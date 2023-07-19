import numpy as np
import jax.numpy as jnp
from jax import nn, jacfwd, jit, vmap, lax
from functools import partial
import time

# import MCMC library
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC

import matplotlib.pyplot as plt

# class that implements standard RNN
class RNN():

    def __init__(self, n_species, n_metabolites, n_controls, n_hidden, N,
                 alpha_0=1., param_0=1.):

        # store dimensions
        self.n_species = n_species
        self.n_metabolites = n_metabolites
        self.n_controls = n_controls
        self.n_hidden = n_hidden
        self.N = N
        self.alpha_0 = alpha_0
        self.param_0 = param_0

        # determine indeces of species, metabolites and controls
        self.n_inp = n_species + n_metabolites + n_controls
        self.s_inds = np.array([False]*self.n_inp)
        self.m_inds = np.array([False]*self.n_inp)
        self.o_inds = np.array([False]*self.n_inp)
        self.u_inds = np.array([False]*self.n_inp)

        self.s_inds[:n_species] = True
        self.m_inds[n_species:n_species+n_metabolites] = True
        self.o_inds[:n_species+n_metabolites] = True
        self.u_inds[n_species+n_metabolites:] = True

        # metabolites refers in general to any predicted variable other than species
        self.n_out = n_species + n_metabolites

        # determine shapes of weights/biases = [Whh,bhh,Wih, Who,bho, h0]
        self.shapes = [[n_hidden, n_hidden], [n_hidden], [n_hidden, n_species+n_metabolites+2*n_controls],
                  [n_species+n_metabolites, n_hidden], [n_species+n_metabolites], [n_hidden]]
        self.k_params = []
        self.n_params = 0
        for shape in self.shapes:
            self.k_params.append(self.n_params)
            self.n_params += np.prod(shape)
        self.k_params.append(self.n_params)

        # initialize parameters
        self.params = np.zeros(self.n_params)
        for k1,k2,shape in zip(self.k_params[:-1], self.k_params[1:-1], self.shapes[:-1]):
            if len(shape)>1:
                stdv = self.param_0/np.sqrt(np.prod(shape))
            self.params[k1:k2] = np.random.uniform(0., stdv, k2-k1)
        self.Ainv = None
        self.a = 1e-4
        self.b = 1e-4

        ### define jit compiled functions ###

        # batch prediction
        self.forward_batch = jit(vmap(self.forward, in_axes=(None, 0)))

        # jit compile gradient w.r.t. params
        self.G  = jit(jacfwd(self.forward_batch))
        self.Gi = jit(jacfwd(self.forward))

        # jit compile Newton update direction computation
        def NewtonStep(G, g, alpha, Beta):
            # compute hessian
            A = jnp.diag(alpha) + jnp.einsum('ntki,kl,ntlj->ij', G, Beta, G)
            # solve for Newton step direction
            d = jnp.linalg.solve(A, g)
            return d
        self.NewtonStep = jit(NewtonStep)

        # jit compile inverse Hessian computation step
        def Ainv_next(G, Ainv, BetaInv):
            GAinv = G@Ainv
            Ainv_step = GAinv.T@jnp.linalg.inv(BetaInv + GAinv@G.T)@GAinv
            Ainv_step = (Ainv_step + Ainv_step.T)/2.
            return Ainv_step
        self.Ainv_next = jit(Ainv_next)

        # jit compile measurement covariance computation
        def compute_yCOV(errors, G, Ainv):
            return jnp.einsum('ntk,ntl->kl', errors, errors) + jnp.einsum('ntki,ij,ntlj->kl', G, Ainv, G)
        self.compute_yCOV = jit(compute_yCOV)

        # jit compile prediction covariance computation
        def compute_predCOV(BetaInv, G, Ainv):
            return BetaInv + jnp.einsum("ntki,ij,ntlj->ntkl", G, Ainv, G)
        self.compute_predCOV = jit(compute_predCOV)

        # jit compile prediction covariance computation
        def compute_predCOVi(BetaInv, Gi, Ainv):
            return BetaInv + jnp.einsum("tki,ij,tlj->tkl", Gi, Ainv, Gi)
        self.compute_predCOVi = jit(compute_predCOVi)

        # jit compile prediction covariance computation
        def compute_searchCOV(Beta, G, Ainv):
            return jnp.eye(Beta.shape[0]) + jnp.einsum("kl,ntli,ij,ntmj->ntkm", Beta, G, Ainv, G)
        self.compute_searchCOV = jit(compute_searchCOV)

        # jit compile prediction covariance computation
        def epistemic_COV(G, Ainv):
            return jnp.einsum("ntki,ij,ntlj->ntkl", G, Ainv, G)
        self.epistemic_COV = jit(epistemic_COV)

    # reshape parameters into weight matrices and bias vectors
    def reshape(self, params):
        # params is a vector = [Whh,bhh,Wih,Who,bho,h0]
        return [np.reshape(params[k1:k2], shape) for k1,k2,shape in zip(self.k_params, self.k_params[1:], self.shapes)]

    # per-sample prediction
    def forward(self, params, sample):
        return self.output(params, sample[:,self.s_inds], sample[:,self.m_inds], sample[:,self.u_inds])

    def output(self, params, s, m, u):
        # reshape params
        Whh,bhh,Wih,Who,bho,h0 = self.reshape(params)
        params = [Whh,bhh,Wih,Who,bho]

        # define rnn
        rnn_ctrl = partial(self.rnn_cell, params, u)

        # define initial value
        init = (0, h0, s[0], m[0])

        # per-example predictions
        carry, out = lax.scan(rnn_ctrl, init, xs=u[1:])
        return out

    # RNN cell
    def rnn_cell(self, params, u, carry, inp):
        # unpack carried values
        t, h, s, m = carry

        # params is a vector = [Whh,bhh,Wih,Who,bho]
        Whh, bhh, Wih, Who, bho = params

        # concatenate inputs
        i = jnp.concatenate((s, m, u[t], u[t+1]))

        # update hidden vector
        h = nn.leaky_relu(Whh@h + Wih@i + bhh)

        # predict output
        o = Who@h + bho
        s, m = o[:len(s)], o[len(s):]

        # return carried values and slice of output
        return (t+1, h, s, m), o

    # fit to data
    def fit(self, data, lr=1e-2, map_tol=1e-3, evd_tol=1e-3,
            patience=3, max_fails=3):
        passes = 0
        fails  = 0
        # fit until convergence of evidence
        previdence = -np.inf
        evidence_converged = False
        epoch = 0
        best_evidence_params = np.copy(self.params)
        best_params = np.copy(self.params)

        while not evidence_converged:

            # update hyper-parameters
            self.update_hypers(data)

            # use Newton descent to determine parameters
            prev_loss = np.inf

            # fit until convergence of NLP
            converged = False
            while not converged:
                # forward passs
                outputs = self.forward_batch(self.params, data)
                errors  = np.nan_to_num(outputs - data[:, 1:, self.o_inds])
                residuals = np.sum(errors)/self.N

                # compute convergence of loss function
                loss = self.compute_loss(errors)
                convergence = (prev_loss - loss) / max([1., loss])
                if epoch%10==0:
                    print("Epoch: {}, Loss: {:.5f}, Residuals: {:.5f}, Convergence: {:5f}".format(epoch, loss, residuals, convergence))

                # stop if less than tol
                if abs(convergence) <= map_tol:
                    # set converged to true to break from loop
                    converged = True
                else:
                    # lower learning rate if convergence is negative
                    if convergence < 0:
                        lr /= 2.
                        # re-try with the smaller step
                        self.params = best_params - lr*d
                    else:
                        # update best params
                        best_params = np.copy(self.params)

                        # update previous loss
                        prev_loss = loss

                        # compute gradients
                        G = self.G(self.params, data)
                        g = np.einsum('ntk,kl,ntli->i', errors, self.Beta, G) + self.alpha*self.params

                        # determine Newton update direction
                        d = self.NewtonStep(G, g, self.alpha, self.Beta)

                        # update parameters
                        self.params -= lr*d

                        # update epoch counter
                        epoch += 1

            # Update Hessian estimation
            G = self.G(self.params, data)
            self.A, self.Ainv = self.compute_precision(G)

            # compute evidence
            evidence = self.compute_evidence(loss)

            # determine whether evidence is converged
            evidence_convergence = (evidence - previdence) / max([1., abs(evidence)])
            print("\nEpoch: {}, Evidence: {:.5f}, Convergence: {:5f}".format(epoch, evidence, evidence_convergence))

            # stop if less than tol
            if abs(evidence_convergence) <= evd_tol:
                passes += 1
                lr *= 2.
            else:
                if evidence_convergence < 0:
                    # reset :(
                    fails += 1
                    self.params = np.copy(best_evidence_params)
                    # Update Hessian estimation
                    G = self.G(self.params, data)
                    self.A, self.Ainv = self.compute_precision(G)

                    # reset evidence back to what it was
                    evidence = previdence
                    # lower learning rate
                    lr /= 2.
                else:
                    passes = 0
                    # otherwise, update previous evidence value
                    previdence = evidence
                    # update measurement covariance
                    self.yCOV = self.compute_yCOV(errors, G, self.Ainv)
                    # update best evidence parameters
                    best_evidence_params = np.copy(self.params)

            # If the evidence tolerance has been passed enough times, return
            if passes >= patience or fails >= max_fails:
                evidence_converged = True

    # update hyper-parameters alpha and Beta
    def update_hypers(self, data):
        if self.Ainv is None:
            self.yCOV = np.zeros([self.n_out, self.n_out])
            for sample in data:
                self.yCOV += np.einsum('tk,tl->kl', np.nan_to_num(sample[:,self.o_inds]), np.nan_to_num(sample[:,self.o_inds]))
            self.yCOV = (self.yCOV + self.yCOV.T)/2.
            # update alpha
            self.alpha = self.alpha_0*jnp.ones_like(self.params)
            # update Beta
            self.Beta = self.N*np.linalg.inv(self.yCOV + 2.*self.b*np.eye(self.n_out))
            self.Beta = (self.Beta + self.Beta.T)/2.
            self.BetaInv = np.linalg.inv(self.Beta)
        else:
            # update alpha
            self.alpha = 1. / (self.params**2 + np.diag(self.Ainv) + 2.*self.a)
            # update beta
            self.Beta = self.N*np.linalg.inv(self.yCOV + 2.*self.b*np.eye(self.n_out))
            self.Beta = (self.Beta + self.Beta.T)/2.
            self.BetaInv = np.linalg.inv(self.Beta)

    # compute loss
    def compute_loss(self, errors):
        return 1/2*(np.einsum('ntk,kl,ntl->', errors, self.Beta, errors) + np.dot(self.alpha*self.params, self.params))

    # compute Precision and Covariance matrices
    def compute_precision(self, G):
        # compute Hessian (precision Matrix)
        A = jnp.diag(self.alpha) + jnp.einsum('ntki,kl,ntlj->ij', G, self.Beta, G)
        A = (A + A.T)/2.

        # compute inverse precision (covariance Matrix)
        Ainv = jnp.diag(1./self.alpha)
        for Gn in G:
            for Gt in Gn:
                Ainv -= self.Ainv_next(Gt, Ainv, self.BetaInv)
        Ainv = (Ainv + Ainv.T)/2.
        return A, Ainv

    # compute the log marginal likelihood
    def compute_evidence(self, loss):
        # compute evidence
        Hessian_eigs = np.linalg.eigvalsh(self.A)
        evidence = self.N/2*np.nansum(np.log(np.linalg.eigvalsh(self.Beta))) + \
                   1/2*np.nansum(np.log(self.alpha)) - \
                   1/2*np.nansum(np.log(Hessian_eigs[Hessian_eigs>0])) - loss
        return evidence

    def fit_MCMC(self, data, num_warmup=1000, num_samples=4000, rng_key=0):
        # define probabilistic model
        def pyro_model():

            # sample from Laplace approximated posterior
            w = numpyro.sample("w",
                               dist.MultivariateNormal(loc=self.params,
                                                       covariance_matrix=self.Ainv))

            # sample from zero-mean Gaussian prior with independent precision priors
            '''with numpyro.plate("params", self.n_params):
                alpha = numpyro.sample("alpha", dist.Exponential(rate=1e-4))
                w = numpyro.sample("w", dist.Normal(loc=0., scale=(1./alpha)**.5))'''

            # sample from zero-mean Gaussian prior with single precision prior
            '''alpha = numpyro.sample("alpha", dist.Exponential(rate=1e-4))
            w = numpyro.sample("w", dist.MultivariateNormal(loc=np.zeros(self.n_params),
                                                            precision_matrix=alpha*np.eye(self.n_params)))'''

            # output of neural network:
            preds = self.forward_batch(w, data)

            # sample model likelihood with max evidence precision matrix
            numpyro.sample("Y",
                           dist.MultivariateNormal(loc=preds, precision_matrix=self.Beta),
                           obs = data[:, 1:, :])

        # init MCMC object with NUTS kernel
        kernel = NUTS(pyro_model, step_size=1.)
        self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

        # warmup
        self.mcmc.warmup(random.PRNGKey(rng_key), init_params=self.params)

        # run MCMC
        self.mcmc.run(random.PRNGKey(rng_key), init_params=self.params)

        # save posterior params
        self.posterior_params = np.array(self.mcmc.get_samples()['w'])

    # function to predict metabolites and variance
    def predict(self, data):
        # make point predictions
        preds = nn.relu(self.forward_batch(self.params, data))

        # include known initial conditions
        preds = np.concatenate((data[:, 0:1, self.o_inds], preds), 1)

        # compute sensitivities
        G = self.G(self.params, data)

        # compute covariances
        COV = self.compute_predCOV(self.BetaInv, G, self.Ainv)

        # pull out standard deviations
        get_diag = vmap(vmap(jnp.diag, (0,)), (0,))
        stdvs = np.sqrt(get_diag(COV))
        stdvs = np.concatenate((np.zeros([preds.shape[0], 1, preds.shape[-1]]), stdvs), 1)

        return preds, stdvs, COV

    # function to predict metabolites and variance
    def predict_point(self, data):
        # make point predictions
        preds = nn.relu(self.forward_batch(self.params, data))

        # include known initial conditions
        preds = np.concatenate((data[:, 0:1, self.o_inds], preds), 1)

        return preds

    # function to predict from posterior samples
    def predict_MCMC(self, data):
        # make point predictions
        preds = jit(vmap(lambda params: nn.relu(self.forward_batch(params, data)), (0,)))(self.posterior_params)

        # take mean and standard deviation
        stdvs = np.sqrt(np.diag(self.BetaInv) + np.var(preds, 0))
        preds = np.mean(preds, 0)

        # include known initial conditions
        preds = np.concatenate((data[:, 0:1, self.o_inds], preds), 1)
        stdvs = np.concatenate((np.zeros([preds.shape[0], 1, preds.shape[-1]]), stdvs), 1)

        return preds, stdvs

    # function to predict metabolites and variance
    def conditioned_stdv(self, data, Ainv):

        # compute sensitivities
        G = self.G(self.params, data)

        # compute covariances
        COV = self.epistemic_COV(G, Ainv)

        # pull out standard deviations
        get_diag = vmap(vmap(jnp.diag, (0,)), (0,))
        stdvs = np.sqrt(get_diag(COV))
        stdvs = np.concatenate((np.zeros([data.shape[0], 1, stdvs.shape[-1]]), stdvs), 1)

        return stdvs

    # return indeces of optimal samples
    def search(self, data, objective, scaler, N,
               P=None, batch_size=512, explore=1e-3, max_explore=1e3):

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # compute objective (f: R^[n_t, n_o] -> R) in batches
        if P is not None:
            objective_batch = jit(vmap(lambda pred, obj_params: objective(scaler.inverse_transform(pred), obj_params)))
        else:
            objective_batch = jit(vmap(lambda pred, obj_params: objective(scaler.inverse_transform(pred))))
            P = jnp.zeros(n_samples)

        objectives = []
        for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
            # make predictions on data
            # use true values only as a sanity check -->
            #objectives.append(objective_batch(data[:,:,self.o_inds][batch_inds]))
            preds = self.predict_point(data[batch_inds])
            objectives.append(objective_batch(preds, P[batch_inds]))
        objectives = jnp.concatenate(objectives).ravel()
        print("Top 5 predicted outcomes: ", jnp.sort(objectives)[::-1][:5])

        if explore <= 0.:
            print("Pure exploitation, returning N max objective experiments")
            return np.array(jnp.argsort(objectives)[::-1][:N])

        # initialize conditional parameter covariance
        Ainv_q = jnp.copy(self.Ainv)

        # initialize with sample that maximizes objective
        best_experiments = [np.argmax(objectives).item()]
        print(f"Picked experiment {len(best_experiments)} out of {N}")
        # update parameter covariance
        Gi = self.Gi(self.params, data[best_experiments[-1]])
        # COVi = self.compute_predCOVi(self.BetaInv, Gi, Ainv_q)
        # for Gt, COVt in zip(Gi, COVi):
        #     Ainv_q -= self.Ainv_next(Gt, Ainv_q, COVt)
        for Gt in Gi:
            Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)

        # define batch function to compute utilities over all samples
        utility_batch = jit(vmap(self.utility))

        # search for new experiments until find N
        while len(best_experiments) < N:

            # compute information acquisition function
            f_I = []
            for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
                # predCOV has shape [n_samples, n_time, n_out, n_out]
                predCOV = self.compute_searchCOV(self.Beta, self.G(self.params, data[batch_inds]), Ainv_q)
                f_I.append(utility_batch(predCOV))
            f_I = jnp.concatenate(f_I).ravel()

            # select next point
            w = 0.
            while jnp.argmax(objectives + w*f_I) in best_experiments and w < max_explore:
                w += explore
            utilities = objectives + w*f_I
            print("Exploration weight set to: {:.3f}".format(w))
            print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])

            # accept if unique
            exp = jnp.argmax(utilities)
            if exp not in best_experiments:
                best_experiments += [exp.item()]
                print(f"Picked experiment {len(best_experiments)} out of {N}")

                # if have enough selected experiments, return
                if len(best_experiments) == N:
                    return best_experiments
                else:
                    # update parameter covariance given selected condition
                    Gi = self.Gi(self.params, data[best_experiments[-1]])
                    # COVi = self.compute_predCOVi(self.BetaInv, Gi, Ainv_q)
                    # for Gt, COVt in zip(Gi, COVi):
                    #     Ainv_q -= self.Ainv_next(Gt, Ainv_q, COVt)
                    for Gt in Gi:
                        Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)
            else:
                print("WARNING: Did not select desired number of conditions")
                return best_experiments

    # compute utility of each experiment
    def utility(self, predCOV):
        # predicted objective + log det of prediction covariance over time series
        # predCOV has shape [n_time, n_out, n_out]
        # log eig predCOV has shape [n_time, n_out]
        # det predCOV has shape [n_time]
        return jnp.mean(jnp.nansum(jnp.log(jnp.linalg.eigvalsh(predCOV)), -1))

    # return indeces of optimal samples
    def fast_search(self, data, objective, scaler, N,
               batch_size=512, explore = .5, max_explore = 1e3):

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # compute objective (f: R^[n_t, n_o, w_exp] -> R) in batches
        objective_batch = jit(vmap(lambda pred, stdv, explore: objective(scaler.inverse_transform(pred),
                                                                         scaler.inverse_transform(stdv),
                                                                         explore), (0,0,None)))

        # initialize search with pure exploitation
        objectives = []
        all_preds  = []
        for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
            # make predictions on data
            preds = self.predict_point(data[batch_inds])
            # evaluate objectives with zero uncertainty
            objectives.append(objective_batch(preds, 0.*preds, 1.))
            # save predictions so that they're only evaluated once
            all_preds.append(preds)
        objectives = jnp.concatenate(objectives)
        print("Top 5 utilities: ", jnp.sort(objectives)[::-1][:5])

        if explore <= 0.:
            print("Pure exploitation, returning N max objective experiments")
            return np.array(jnp.argsort(objectives)[::-1][:N])

        # initialize with sample that maximizes objective
        best_experiments = [np.argmax(objectives).item()]
        print(f"Picked experiment {len(best_experiments)} out of {N}")

        # initialize conditioned parameter covariance
        Ainv_q = jnp.copy(self.Ainv)
        Gi = self.Gi(self.params, data[np.argmax(objectives)])
        # update conditioned parameter covariance
        for Gt in Gi:
            Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)

        # search for new experiments until find N
        eval_utilities = True
        while len(best_experiments) < N:

            # compute utilities in batches to avoid memory problems
            utilities = []
            for preds, batch_inds in zip(all_preds, np.array_split(np.arange(n_samples), n_samples//batch_size)):
                stdvs = self.conditioned_stdv(data[batch_inds], Ainv_q)
                utilities.append(objective_batch(preds, stdvs, explore))
            utilities = jnp.concatenate(utilities)
            print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])

            # sort utilities from best to worst
            exp_sorted = jnp.argsort(utilities)[::-1]
            for exp in exp_sorted:
                # accept if unique
                if exp not in best_experiments:
                    best_experiments += [exp.item()]
                    # compute sensitivity to sample
                    Gi = self.Gi(self.params, data[exp])
                    # update conditioned parameter covariance
                    for Gt in Gi:
                        Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)
                    print(f"Picked experiment {len(best_experiments)} out of {N}")
                    if eval_utilities:
                        break
                    # IF already exceed max exploration
                    # AND have enough selected experiments, return
                    if len(best_experiments) == N:
                        return best_experiments

                # increase exploration if not unique
                elif explore < max_explore:
                    explore *= 2.
                    print("Increased exploration rate to {:.3f}".format(explore))
                    break
                else:
                    # if the same experiment was picked twice at the max
                    # exploration rate, do not re-evaluate utilities
                    eval_utilities = False

        return best_experiments

    # return indeces of optimally informative samples
    def explore(self, data, N, batch_size=512):

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # initialize FIM
        BFIM = self.A
        best_experiments = []

        # define batch function to compute utilities over all samples
        # FIM_batch = jit(self.FIM_batch)
        utility_batch = jit(vmap(self.explore_utility, (0,None)))

        # search for new experiments until find N
        while len(best_experiments) < N:

            # compute utilities in batches to avoid memory problems
            utilities = []
            for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
                utilities.append(utility_batch(data[batch_inds], BFIM))
            utilities = jnp.concatenate(utilities)
            print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])

            # sort utilities from best to worst
            exp_sorted = jnp.argsort(utilities)[::-1]
            for exp in exp_sorted:
                # accept if unique
                if exp not in best_experiments:
                    best_experiments += [exp.item()]
                    # scale = N predicted time points
                    BFIM += self.FIM(data[exp])
                    print(f"Picked experiment {len(best_experiments)} out of {N}")
                    break

        return best_experiments

    # compute utility of each experiment
    def explore_utility(self, sample, BFIM):
        # unfortunately jnp eigvals'h' function does not work properly yet
        return jnp.nansum(jnp.log(jnp.linalg.eigvalsh(BFIM + self.FIM(sample))))

# class that implements microbiome RNN (miRNN)
class miRNN(RNN):

    # RNN cell
    def rnn_cell(self, params, u, carry, inp):
        # unpack carried values
        t, h, s, m = carry

        # params is a vector = [Whh,bhh,Wih,Who,bho]
        Whh, bhh, Wih, Who, bho = params

        # concatenate inputs
        i = jnp.concatenate((s, m, u[t], u[t+1]))

        # update hidden vector
        h = nn.leaky_relu(Whh@h + Wih@i + bhh)

        # predict output
        zeros_mask = jnp.concatenate((jnp.array(s>0, float), jnp.ones(m.shape)))
        o = zeros_mask*(Who@h + bho)
        s, m = o[:len(s)], o[len(s):]

        # return carried values and slice of output
        return (t+1, h, s, m), o

class miRNNA(miRNN):

    # return indeces of optimal samples
    def search(self, data, objective, scaler, N,
               batch_size=512, explore = 1e-3, max_explore = 1e3):

        # jit compile FIM function using current parameters
        def FIM(sample):
            # compute sensitivities
            Gi = self.Gi(self.params, sample)

            # compute covariances
            COVi = self.BetaInv + jnp.einsum("tki,i,tli->tkl", Gi, jnp.diag(self.Ainv), Gi)

            # compute FIM diagonal
            FIMi = jnp.einsum("tki,tkl,tli->i", Gi, jnp.linalg.inv(COVi), Gi)
            return FIMi
        self.FIM = jit(FIM)

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # compute objective (f: [n_t, n_o] -> R) in batches
        objective_batch = jit(vmap(lambda pred: objective(scaler.inverse_transform(pred))))
        objectives = []
        for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
            # make predictions on data
            # use true values only as a sanity check -->
            #objectives.append(objective_batch(data[:,:,self.o_inds][batch_inds]))
            preds = self.predict_point(data[batch_inds])
            objectives.append(objective_batch(preds))
        objectives = jnp.concatenate(objectives)
        if explore <= 0.:
            print("Pure exploitation, returning N max objective experiments")
            return np.array(jnp.argsort(objectives)[::-1][:N])

        # could compute FIM_i for each data point first
        # FIMS = vmap(self.FIM)(data, COV) # <-- too memory intensive

        # initialize with sample that maximizes objective
        best_experiments = [np.argmax(objectives).item()]
        BFIM = jnp.diag(self.A) + self.FIM(data[np.argmax(objectives)])
        print(f"Picked experiment {len(best_experiments)} out of {N}")

        # define batch function to compute utilities over all samples
        utility_batch = jit(vmap(self.utility, (0,0,None,None)))

        # search for new experiments until find N
        eval_utilities = True
        while len(best_experiments) < N:

            # compute utilities in batches to avoid memory problems
            utilities = []
            for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
                utilities.append(utility_batch(objectives[batch_inds], data[batch_inds], explore, BFIM))
            utilities = jnp.concatenate(utilities)
            print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])

            # sort utilities from best to worst
            exp_sorted = jnp.argsort(utilities)[::-1]
            for exp in exp_sorted:
                # accept if unique
                if exp not in best_experiments:
                    best_experiments += [exp.item()]
                    BFIM += self.FIM(data[exp])
                    print(f"Picked experiment {len(best_experiments)} out of {N}")
                    if eval_utilities:
                        break
                    # IF already exceed max exploration
                    # AND have enough selected experiments, return
                    if len(best_experiments) == N:
                        return best_experiments
                # increase exploration if not unique
                elif explore < max_explore:
                    explore = np.clip(10*explore, 0., max_explore)
                    print("Increased exploration rate to {:.3f}".format(explore))
                    break
                else:
                    # if the same experiment was picked twice at the max
                    # exploration rate, do not re-evaluate utilities
                    eval_utilities = False

        return best_experiments

    # compute utility of each experiment
    def utility(self, objective, sample, explore, BFIM):
        return objective/explore + jnp.nansum(jnp.log(BFIM + self.FIM(sample)))

    # return indeces of optimally informative samples
    def explore(self, data, N, batch_size=512):

        # jit compile FIM function using current parameters
        def FIM(sample):
            # compute sensitivities
            Gi = self.Gi(self.params, sample)

            # compute covariances
            COVi = self.BetaInv + jnp.einsum("tki,i,tli->tkl", Gi, jnp.diag(self.Ainv), Gi)

            # compute FIM diagonal
            FIMi = jnp.einsum("tki,tkl,tli->i", Gi, jnp.linalg.inv(COVi), Gi)
            return FIMi
        self.FIM = jit(FIM)

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # initialize FIM
        BFIM = jnp.diag(self.A)
        best_experiments = []

        # define batch function to compute utilities over all samples
        # FIM_batch = jit(self.FIM_batch)
        utility_batch = jit(vmap(self.explore_utility, (0,None)))

        # search for new experiments until find N
        eval_utilities = True
        while len(best_experiments) < N:

            # compute utilities in batches to avoid memory problems
            utilities = []
            for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
                utilities.append(utility_batch(objectives[batch_inds], data[batch_inds], explore, BFIM))
            utilities = jnp.concatenate(utilities)
            print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])

            # sort utilities from best to worst
            exp_sorted = jnp.argsort(utilities)[::-1]
            for exp in exp_sorted:
                # accept if unique
                if exp not in best_experiments:
                    best_experiments += [exp.item()]
                    BFIM += self.FIM(data[exp])
                    print(f"Picked experiment {len(best_experiments)} out of {N}")
                    if eval_utilities:
                        break
                    # IF already exceed max exploration
                    # AND have enough selected experiments, return
                    if len(best_experiments) == N:
                        return best_experiments
                # increase exploration if not unique
                elif explore < max_explore:
                    explore = np.clip(10*explore, 0., max_explore)
                    print("Increased exploration rate to {:.3f}".format(explore))
                    break
                else:
                    # if the same experiment was picked twice at the max
                    # exploration rate, do not re-evaluate utilities
                    eval_utilities = False

        return best_experiments

    # compute utility of each experiment
    def explore_utility(self, sample, BFIM):
        # unfortunately jnp eigvals'h' function does not work properly yet
        return jnp.nansum(jnp.log(BFIM + self.FIM(sample)))
