import os
import multiprocessing
from functools import partial

# set number of processors
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import numpy as np
# import libraries to compute gradients
from jax import pmap, random
from jax.experimental.ode import odeint
from jax.nn import sigmoid

# matrix math
from .linalg import *

# import model class
from .model import MODEL

# Function to process dataframes
def process_df(df, species, resources):
    # store measured datasets for quick access
    data = []
    for treatment, comm_data in df.groupby("Experiments"):
        # make sure comm_data is sorted in chronological order
        comm_data.sort_values(by='Time', ascending=True, inplace=True)

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, np.float32)

        # pull species data
        y_species = np.array(comm_data[species].values, np.float32)

        # pull resources data
        y_resources = np.array(comm_data[resources].values, np.float32)

        # concatenate measured values
        y_measured = np.concatenate((y_species, y_resources), 1)

        # append t_eval and y_measured to data list
        data.append([treatment, t_eval, y_measured])

    # loop over each sample in dataset
    data_dict = {}
    for treatment, t_eval, y_measured in data:

        # enter data into dictionary
        if len(t_eval) not in data_dict.keys():
            # create new entry
            data_dict[len(t_eval)] = [t_eval, np.expand_dims(y_measured, 0)]
        else:
            # append to previous entries
            _, Y_measured_ = data_dict[len(t_eval)]
            data_dict[len(t_eval)] = [t_eval, np.concatenate((Y_measured_, np.expand_dims(y_measured, 0)))]

    return data, data_dict

class CR(MODEL):
    def __init__(self, dataframe, species, resources, r0=[], alpha_0=1., verbose=True, rng_key=123):

        # set rng key
        rng_key = random.PRNGKey(rng_key)

        # number of available devices for parallelizing code
        self.n_devices = multiprocessing.cpu_count()

        # initial conditions of unobserved resources
        self.r0 = jnp.atleast_1d(jnp.array(r0, dtype=float))

        # initial regularization
        self.alpha_0 = alpha_0

        # dimensions
        self.n_s = len(species)
        self.n_m = len(resources) + len(self.r0)

        # number of observed variables
        self.n_obs = self.n_s + self.n_m - len(self.r0)

        # set up data
        self.dataset, self.data_dict = process_df(dataframe, species, resources)

        # for additional output messages
        self.verbose = verbose

        ### JIT compiled helper functions to integrate ODEs in parallel ###

        # batch evaluation of system
        self.batch_ODE = pmap(self.runODE, in_axes=(None, 0, None))

        # batch evaluation of sensitivity equations
        self.batch_ODEZ = pmap(self.runODEZ, in_axes=(None, 0, None))

    # initialize model parameters
    def init_params(self, rng_key):

        # set rng key
        self.rng_key = random.PRNGKey(rng_key)

        # initialize consumer resource parameters

        # species efficiency
        f = np.ones(self.n_s)

        # death rate
        d = np.log2(np.ones(self.n_s)/5.)

        # log of [C]_ij = rate that species j consumes resource i
        C = np.log2(random.uniform(self.rng_key, shape=(self.n_m, self.n_s), minval=0., maxval=2.))

        # log of [P]_ij = rate that species j produces resource i
        P = np.log2(random.uniform(self.rng_key, shape=(self.n_m, self.n_s), minval=0., maxval=.05))

        # concatenate parameter initial guess
        self.params = (f, d, C, P)

        # determine shapes of parameters
        self.shapes = []
        self.k_params = []
        self.n_params = 0
        for param in self.params:
            self.shapes.append(param.shape)
            self.k_params.append(self.n_params)
            self.n_params += param.size
        self.k_params.append(self.n_params)

        # set prior so that C and P are sparse
        C_0 = np.zeros_like(C)
        P_0 = np.zeros_like(P)

        # concatenate prior mean
        prior = [f, d, C_0, P_0]
        self.prior = np.concatenate([p.ravel() for p in prior])

        # hyper-prior parameters
        self.a = 1e-4
        self.b = 1e-4

        # initial condition of grad system w.r.t. params
        self.Z0 = [np.zeros([self.n_s + self.n_m] + list(param.shape)) for param in self.params]

    # compute residual between parameter estimate and prior
    def param_res(self, params):
        # only take exp of strictly positive parameters
        param_copy = np.copy(params)
        param_copy[self.n_s:] = np.exp2(params[self.n_s:])

        # residuals
        res = param_copy - self.prior

        return res

    # define neural consumer resource model
    @partial(jit, static_argnums=0)
    def system(self, x, t, params):

        # species
        s = x[:self.n_s]

        # resources
        r = x[self.n_s:]

        # unpack params
        f, d, C, P = params

        # efficiencies must be between 0 and 1
        f = sigmoid(f)

        # take exp of strictly positive params
        d = jnp.exp2(d)
        C = jnp.exp2(C)
        P = jnp.exp2(P)

        # rate of change of species
        dsdt = s * f * (C.T @ r - d)

        # rate of change of log of resources
        drdt = P @ s - r * (C @ s)

        return jnp.concatenate((dsdt, drdt))
