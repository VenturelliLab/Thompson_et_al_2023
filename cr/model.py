import os
from scipy.io import savemat, loadmat
import multiprocessing
from functools import partial

# set number of processors
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import numpy as np
import pandas as pd 
# import libraries to compute gradients
from jax import jacfwd, vmap
from jax.experimental.ode import odeint

# matrix math
from .linalg import *

class MODEL:

    # augmented system for forward sensitivity equations
    def __init__(self):
        self.species = None
        self.mediators = None
        self.a = 1e-4
        self.b = 1e-4
        self.n_devices = None
        self.n_params = None
        self.data_dict = None
        self.dataset = None
        self.params = None
        self.verbose = None
        self.r0 = None
        self.prior = None
        self.n_cr_params = None
        self.shapes = None
        self.k_params = None
        self.Z0 = None
        self.system = None
        self.n_m = None
        self.n_obs = None
        self.Y0 = None
        self.n_s = None

    @partial(jit, static_argnums=0)
    def aug_system(self, aug_x, t, params):

        # unpack augmented state
        x = aug_x[0]
        Z = aug_x[1:]

        # time derivative of state
        dxdt = self.system(x, t, params)

        # system Jacobian
        Jx_i = jacfwd(self.system, 0)(x, t, params)

        # time derivative of parameter sensitivity
        dZdt = [jnp.einsum("ij,j...->i...", Jx_i, Z_i) + Jp_i for Z_i, Jp_i in
                zip(Z, jacfwd(self.system, 2)(x, t, params))]

        return dxdt, *dZdt

    # integrate system of equations
    @partial(jit, static_argnums=0)
    def runODE(self, t_eval, x, params):
        # get initial condition
        x_ic = x[0]
        return odeint(self.system, jnp.concatenate((x_ic, self.r0)), t_eval, params)

    # integrate forward sensitivity equations
    @partial(jit, static_argnums=0)
    def runODEZ(self, t_eval, x, params):
        # get initial condition
        x_ic = x[0]
        return odeint(self.aug_system, [jnp.concatenate((x_ic, self.r0)), *self.Z0], t_eval, params)

    # reshape parameters into weight matrices and bias vectors
    def reshape(self, params):
        return [np.array(np.reshape(params[k1:k2], shape), dtype=np.float32) for k1, k2, shape in
                zip(self.k_params, self.k_params[1:], self.shapes)]

    # solve Ax = b for x
    @partial(jit, static_argnums=0)
    def NewtonStep(self, A, g):
        return jnp.linalg.solve(A, g)

    # fit to data
    def fit(self, lr=1e-1, map_tol=1e-3, evd_tol=1e-3, patience=3, max_fails=3, rng_key=123):
        passes = 0
        fails  = 0
        # fit until convergence of evidence
        previdence = -np.inf
        evidence_converged = False
        epoch = 0

        # params as a flattened vector
        self.init_params(rng_key)
        params = np.concatenate([p.ravel() for p in self.params])

        # set best parameters
        best_evidence_params = np.copy(params)
        best_params = np.copy(params)

        while not evidence_converged:

            # update hyper-parameters
            if epoch == 0:
                self.init_hypers(self.alpha_0)
            else:
                self.update_hypers()

            # use Newton descent to determine parameters
            prev_loss = np.inf

            # fit until convergence of NLP
            converged = False
            while not converged:
                # forward passs
                loss = self.objective(params)
                convergence = (prev_loss - loss) / max([1., loss])
                if epoch%1==0:
                    print("Epoch: {}, Loss: {:.5f}, Residuals: {:.5f}, Convergence: {:5f}".format(epoch, loss, self.RES, convergence))

                # stop if less than tol
                if abs(convergence) <= map_tol:
                    # set converged to true to break from loop
                    converged = True
                else:
                    # lower learning rate if convergence is negative
                    if convergence < 0:
                        lr /= 2.
                        # re-try with the smaller step
                        params = best_params - lr*d
                    else:
                        # update best params
                        best_params = np.copy(params)

                        # update previous loss
                        prev_loss = loss

                        # compute gradients
                        A = self.hessian(params)
                        g = self.jacobian_fwd(params)

                        # determine Newton update direction
                        d = self.NewtonStep(A, g)

                        # update parameters
                        params -= lr*d

                        # update epoch counter
                        epoch += 1

            # Update Hessian estimation
            self.params = self.reshape(params)
            self.update_precision()
            self.update_covariance()

            # compute evidence
            self.update_evidence()

            # determine whether evidence is converged
            evidence_convergence = (self.evidence - previdence) / max([1., abs(self.evidence)])
            print("\nEpoch: {}, Evidence: {:.5f}, Convergence: {:5f}".format(epoch, self.evidence, evidence_convergence))

            # stop if less than tol
            if abs(evidence_convergence) <= evd_tol:
                passes += 1
                lr *= 2.
            else:
                if evidence_convergence < 0:
                    # reset :(
                    fails += 1
                    params = np.copy(best_evidence_params)

                    # Update Hessian estimation
                    self.update_precision()
                    self.update_covariance()

                    # reset evidence back to what it was
                    self.evidence = previdence

                    # lower learning rate
                    lr /= 2.
                else:
                    passes = 0
                    # otherwise, update previous evidence value
                    previdence = self.evidence
                    # update best evidence parameters
                    best_evidence_params = np.copy(params)

            # If the evidence tolerance has been passed enough times, return
            if passes >= patience or fails >= max_fails:
                evidence_converged = True

        # once converged set parameters
        self.params = self.reshape(params)

    def init_hypers(self, alpha_0):

        # count number of samples
        self.N = 0

        # init series
        series = 0.

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # count effective number of uncorrelated observations
            k = 0  # number of outputs
            for series in Y_measured.T:
                # check if there is any variation in the series
                if np.std(series) > 0:
                    # count number of outputs that vary over time
                    k += 1
            assert k > 0, f"There are no time varying outputs in sample {treatment}"

            # adjust N to account for unmeasured outputs
            self.N += (len(series) - 1) * k / self.n_s

        # init output precision
        self.Beta = np.eye(self.n_obs)
        self.BetaInv = np.eye(self.n_obs)

        # initial guess of parameter precision
        self.alpha = alpha_0
        self.Alpha = alpha_0 * np.ones(self.n_params)

        if self.verbose:
            print("Total samples: {:.0f}, Number of parameters: {:.0f}, Initial regularization: {:.2e}".format(self.N,
                                                                                                               self.n_params,
                                                                                                               self.alpha))

    # EM algorithm to update hyperparameters
    def update_hypers(self):
        print("Updating hyper-parameters...")

        # init yCOV
        yCOV = 0.

        # loop over each sample in dataset
        # for treatment, t_eval, Y_measured in self.dataset:
        for n_t, (t_eval, Y_batch) in self.data_dict.items():
            # divide into batches
            n_samples = Y_batch.shape[0]
            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):
                # batches of outputs, initial condition sensitivity, parameter sensitivity
                out_b, *Z_b = self.batch_ODEZ(t_eval, Y_batch[batch_inds], self.params)

                # collect gradients and reshape
                Z_b = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], Z_i.shape[2], -1) for Z_i in Z_b], -1)

                # loop over the batched outputs
                for output, G, Y_measured in zip(np.array(out_b), np.array(Z_b), Y_batch[batch_inds]):

                    # Determine SSE of log of Y
                    Y_error = np.nan_to_num(output[1:, :self.n_obs]) - Y_measured[1:]
                    yCOV += yCOV_next(Y_error, G[1:, :self.n_obs, :], self.Ainv)

        ### M step: update hyper-parameters ###

        # maximize complete data log-likelihood w.r.t. alpha and beta
        Ainv_ii = np.diag(self.Ainv)
        params = np.concatenate([p.ravel() for p in self.params])
        self.alpha = self.n_params / (np.sum(self.param_res(params) ** 2) + np.sum(Ainv_ii) + 2. * self.a)
        # self.Alpha = self.alpha*np.ones(self.n_params)
        self.Alpha = 1. / (self.param_res(params) ** 2 + Ainv_ii + 2. * self.a)

        # update output precision
        self.Beta = self.N * np.linalg.inv(yCOV + 2. * self.b * np.eye(self.n_obs))
        self.Beta = (self.Beta + self.Beta.T) / 2.
        self.BetaInv = np.linalg.inv(self.Beta)

        if self.verbose:
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(self.N, self.alpha))

    def objective(self, params):

        # compute negative log posterior (NLP)
        self.NLP = np.sum(self.Alpha * self.param_res(params) ** 2) / 2.
        # compute residuals
        self.RES = 0.

        # reshape params and convert to JAX tensors
        params = self.reshape(params)

        # loop over each sample in dataset
        # for treatment, t_eval, Y_measured in self.dataset:
        for n_t, (t_eval, Y_batch) in self.data_dict.items():
            # divide into batches
            n_samples = Y_batch.shape[0]
            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):
                batch_output = np.array(self.batch_ODE(t_eval, Y_batch[batch_inds], params))
                # loop over the batched outputs
                for output, y_measured in zip(batch_output, Y_batch[batch_inds]):

                    # Determine error
                    Y_error = np.nan_to_num(output[1:, :self.n_obs]) - y_measured[1:]

                    # Determine SSE and gradient of SSE
                    self.NLP += np.einsum('tk,kl,tl->', Y_error, self.Beta, Y_error) / 2.
                    self.RES += np.sum(Y_error) / self.N

        # return NLP
        return self.NLP

    # gradient of NLP w.r.t. parameters
    def jacobian_fwd(self, params):

        # compute gradient of negative log posterior
        grad_NLP = self.Alpha * self.param_res(params)

        # reshape params and convert to JAX tensors
        params = self.reshape(params)

        # loop over each sample in dataset
        # for treatment, t_eval, Y_measured in self.dataset:
        for n_t, (t_eval, Y_batch) in self.data_dict.items():
            # divide into batches
            n_samples = Y_batch.shape[0]
            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):
                # batches of outputs, initial condition sensitivity, parameter sensitivity
                out_b, *Z_b = self.batch_ODEZ(t_eval, Y_batch[batch_inds], params)

                # collect gradients and reshape
                Z_b = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], Z_i.shape[2], -1) for Z_i in Z_b], -1)

                # loop over the batched outputs
                for output, G, Y_measured in zip(np.array(out_b), np.array(Z_b), Y_batch[batch_inds]):

                    # determine error
                    Y_error = np.nan_to_num(output[1:, :self.n_obs]) - Y_measured[1:]

                    # sum over time and outputs to get gradient w.r.t params
                    grad_NLP += eval_grad_NLP(Y_error, self.Beta, G[1:, :self.n_obs, :])

                    # # compare to finite differences
                    # # d NLP(theta) d_theta_i = [NLP(theta_i + eps) - NLP(theta_i)] / eps
                    # grad_NLP = eval_grad_NLP(Y_error, self.Beta, G[1:, :self.n_obs, :])
                    #
                    # # error with unperturbed parameters
                    # out = np.array(self.runODE(t_eval, Y_measured, r0, params, input))
                    # # undo log transform of mediators
                    # out[:, self.n_s:] = np.exp2(out[:, self.n_s:])
                    # err = out[1:, :self.n_obs] - Y_measured[1:, :self.n_obs]
                    # nll = np.einsum("ti,ij,tj->", err, self.Beta, err) / 2.
                    #
                    #
                    # eps = 1e-3
                    # fd_grad_params = np.zeros_like(grad_NLP)
                    # # tricky because params are weirdly shaped
                    # for i, _ in enumerate(fd_grad_params):
                    #
                    #     # error plus eps
                    #     fd_params = np.concatenate([np.copy(p).ravel() for p in params])
                    #     fd_params[i] += eps
                    #     fd_params = self.reshape(fd_params)
                    #     fd_out = np.array(self.runODE(t_eval, Y_measured, r0, fd_params, input))
                    #     # undo log transform of mediators
                    #     fd_out[:, self.n_s:] = np.exp2(fd_out[:, self.n_s:])
                    #
                    #     fd_err = fd_out[1:, :self.n_obs] - Y_measured[1:, :self.n_obs]
                    #     fd_nll = np.einsum("ti,ij,tj->", fd_err, self.Beta, fd_err)/2.
                    #
                    #     # approximate gradient
                    #     fd_grad_params[i] = (fd_nll - nll)/eps
                    #
                    # # plot
                    # plt.scatter(fd_grad_params, grad_NLP)
                    # plt.show()

        # return gradient of NLP
        return grad_NLP

    def hessian(self, params):

        # reshape params and convert to JAX tensors
        params = self.reshape(params)

        # compute Hessian of NLP
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        # for treatment, t_eval, Y_measured in self.dataset:
        for n_t, (t_eval, Y_batch) in self.data_dict.items():
            # divide into batches
            n_samples = Y_batch.shape[0]
            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):
                # batches of outputs, initial condition sensitivity, parameter sensitivity
                out_b, *Z_b = self.batch_ODEZ(t_eval, Y_batch[batch_inds], params)

                # collect gradients and reshape
                Z_b = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], Z_i.shape[2], -1) for Z_i in Z_b], -1)

                # loop over the batched outputs
                for output, G, Y_measured in zip(np.array(out_b), np.array(Z_b), Y_batch[batch_inds]):

                    # compute Hessian
                    self.A += A_next(G[1:, :self.n_obs, :], self.Beta)

                    # make sure precision is symmetric
                    self.A = (self.A + self.A.T) / 2.

        # return Hessian
        return self.A

    def update_precision(self):

        # compute Hessian of NLP
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        # for treatment, t_eval, Y_measured in self.dataset:
        for n_t, (t_eval, Y_batch) in self.data_dict.items():
            # divide into batches
            n_samples = Y_batch.shape[0]
            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):
                # batches of outputs, initial condition sensitivity, parameter sensitivity
                out_b, *Z_b = self.batch_ODEZ(t_eval, Y_batch[batch_inds], self.params)

                # collect gradients and reshape
                Z_b = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], Z_i.shape[2], -1) for Z_i in Z_b], -1)

                # loop over the batched outputs
                for output, G, Y_measured in zip(np.array(out_b), np.array(Z_b), Y_batch[batch_inds]):

                    # compute Hessian
                    self.A += A_next(G[1:, :self.n_obs, :], self.Beta)

                    # make sure precision is symmetric
                    self.A = (self.A + self.A.T) / 2.

    def update_covariance(self):
        # Approximate / fast method, requires that A is positive definite
        # self.Ainv = compute_Ainv(self.A)

        # compute Hessian of NLP
        self.Ainv = np.diag(1./self.Alpha)

        # loop over each sample in dataset
        # for treatment, t_eval, Y_measured in self.dataset:
        for n_t, (t_eval, Y_batch) in self.data_dict.items():
            # divide into batches
            n_samples = Y_batch.shape[0]
            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):
                # batches of outputs, initial condition sensitivity, parameter sensitivity
                out_b, *Z_b = self.batch_ODEZ(t_eval, Y_batch[batch_inds], self.params)

                # collect gradients and reshape
                Z_b = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], Z_i.shape[2], -1) for Z_i in Z_b], -1)

                # loop over the batched outputs
                for output, G, Y_measured in zip(np.array(out_b), np.array(Z_b), Y_batch[batch_inds]):

                    # compute Hessian
                    for Gi in G[1:, :self.n_obs]:
                        self.Ainv -= Ainv_next(Gi, self.Ainv, self.BetaInv)

                    # make sure precision is symmetric
                    self.Ainv = (self.Ainv + self.Ainv.T) / 2.

        # make sure Ainv is positive definite
        self.Ainv, _ = make_pos_def(self.Ainv, jnp.ones_like(self.Alpha))

    # compute the log marginal likelihood
    def update_evidence(self):

        # compute evidence
        self.evidence = self.N / 2 * log_det(self.Beta) + \
                        1 / 2 * np.nansum(np.log(self.Alpha)) - \
                        1 / 2 * log_det(self.A) - self.NLP

        # print evidence
        if self.verbose:
            print("Evidence {:.3f}".format(self.evidence))

    def callback(self, xk, res=None):
        if self.verbose:
            print("Loss: {:.3f}, Residuals: {:.3f}".format(self.NLP, self.RES))
        return True

    def predict_point(self, x_test, t_eval=None):

        # convert to torch tensors
        t_eval = np.array(t_eval, dtype=np.float32)
        x_test = np.atleast_2d(np.array(x_test, dtype=np.float32))

        # make predictions given initial conditions and evaluation times
        output = np.nan_to_num(self.runODE(t_eval, x_test, self.params))

        return output[:, :self.n_s], output[:, self.n_s:]

    # Define function to make predictions on test data
    def predict(self, x_test, t_eval=None, n_std=1.):

        # integrate forward sensitivity equations
        xYZ = self.runODEZ(t_eval, np.atleast_2d(x_test), self.params)
        output = np.nan_to_num(np.array(xYZ[0]))

        # collect gradients and reshape
        G = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in xYZ[1:]], -1)

        # calculate covariance of each output (dimension = [steps, outputs])
        BetaInv = np.zeros([self.n_s + self.n_m, self.n_s + self.n_m])
        BetaInv[:self.n_obs, :self.n_obs] = self.BetaInv
        covariance = BetaInv + GAinvG(G, self.Ainv)

        # predicted stdv of log y
        get_diag = vmap(jnp.diag, (0,))
        stdv = n_std * np.sqrt(get_diag(covariance))

        return output[:, :self.n_s], stdv[:, :self.n_s], output[:, self.n_s:], stdv[:, self.n_s:]

    # Define function to make predictions on test data
    def predict_df(self, df_test, species, resources):

        # init dataframe with predictions
        df_pred = pd.DataFrame()

        # for each condition
        for exp_name, comm_data in df_test.groupby("Experiments"):

            # make sure comm_data is sorted in chronological order
            comm_data.sort_values(by='Time', ascending=True, inplace=True)
            tspan = np.array(comm_data.Time.values, float)

            # pull just the community data
            output_true = comm_data[list(species)+list(resources)].values

            # determine initial condition
            s0 = output_true[0, :len(species)]
            r0 = output_true[0, len(species):]
            x0 = np.concatenate((s0, r0))

            # predict end-point measured values
            s_pred, s_stdv, r_pred, r_stdv = self.predict(x0, tspan)

            # dataframe for exp
            df_exp = pd.DataFrame()
            df_exp['Experiments'] = [exp_name]*len(tspan)
            df_exp['Time'] = list(tspan)

            for i, variable in enumerate(species):
                df_exp[variable + " true"] = output_true[:, i]
                df_exp[variable + " pred"] = s_pred[:, i]
                df_exp[variable + " stdv"] = s_stdv[:, i]

            for i, variable in enumerate(resources):
                df_exp[variable + " true"] = output_true[:, len(species)+i]
                df_exp[variable + " pred"] = r_pred[:, i]
                df_exp[variable + " stdv"] = r_stdv[:, i]

            df_pred = pd.concat((df_pred, df_exp))

        return df_pred

    def save(self, fname):
        # save model parameters needed to make predictions
        save_dict = {'BetaInv': self.BetaInv, 'Ainv': self.Ainv}

        # save list of params
        for i, p in enumerate(self.params):
            save_dict[f'param_{i}'] = p

        savemat(fname, save_dict)

    def load(self, fname):
        # load model parameters
        load_dict = loadmat(fname)

        # set params
        self.BetaInv = load_dict['BetaInv']
        self.Ainv = load_dict['Ainv']

        # determine number of parameter matrices
        n_items = 1 + max(int(p.split('_')[-1]) for p in load_dict if 'param' in p)

        self.params = []
        for i in range(n_items):
            param = load_dict[f'param_{i}']
            if param.shape[0] > 1:
                self.params.append(param)
            else:
                self.params.append(param.ravel())

        # initial condition of grad system w.r.t. initial latent mediators
        self.Z0 = [np.zeros([self.n_s + self.n_m] + list(param.shape)) for param in self.params]
