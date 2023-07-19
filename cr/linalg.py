from jax import jit
import jax.numpy as jnp

### JIT compiled functions ###

@jit
def steep_sigmoid(x, a=1.):
    return 1./(1. + jnp.exp(-a*x))

### JIT compiled matrix operations ###

@jit
def GAinvG(G, Ainv):
    return jnp.einsum("tki,ij,tlj->tkl", G, Ainv, G)


@jit
def yCOV_next(Y_error, G, Ainv):
    # sum over time dimension
    return jnp.einsum('tk,tl->kl', Y_error, Y_error) + jnp.sum(GAinvG(G, Ainv), 0)

@jit
def yCOV_next_diag(Y_error, G, Ainv):
    # sum over time dimension
    return jnp.einsum('tk,tl->kl', Y_error, Y_error) + jnp.einsum("tki,i,tli->kl", G, Ainv, G)


@jit
def A_next(G, Beta):
    A_n = jnp.einsum('tki, kl, tlj->ij', G, Beta, G)
    A_n = (A_n + A_n.T) / 2.
    return A_n

@jit
def A_next_diag(G, Beta):
    A_n = jnp.einsum('tki, kl, tli->i', G, Beta, G)
    return A_n

# jit compile inverse Hessian computation step
@jit
def Ainv_next(G, Ainv, BetaInv):
    # [G] = [n_out, n_params]
    GAinv = G @ Ainv  # [n_t, n_p]
    Ainv_step = GAinv.T @ jnp.linalg.inv(BetaInv + GAinv @ G.T) @ GAinv
    # Ainv_step = jnp.einsum("ti,tk,kj->ij", GAinv, jnp.linalg.inv(BetaInv + jnp.einsum("ti,ki->tk", GAinv, G)), GAinv)
    Ainv_step = (Ainv_step + Ainv_step.T) / 2.
    return Ainv_step


# jit compile inverse Hessian computation step
@jit
def Ainv_prev(G, Ainv, BetaInv):
    GAinv = G @ Ainv
    Ainv_step = GAinv.T @ jnp.linalg.inv(GAinv @ G.T - BetaInv) @ GAinv
    Ainv_step = (Ainv_step + Ainv_step.T) / 2.
    return Ainv_step


# jit compile function to compute log of determinant of a matrix
@jit
def log_det(A):
    # # using the SVD
    # u,s,v = jnp.linalg.svd(A)
    # return jnp.sum(jnp.log(s))

    # using a Cholesky decomposition
    L = jnp.linalg.cholesky(A)
    return 2 * jnp.sum(jnp.log(jnp.diag(L)))

# approximate inverse of A, where A = LL^T, Ainv = Linv^T Linv
@jit
def compute_Ainv(A):
    Linv = jnp.linalg.inv(jnp.linalg.cholesky(A))
    Ainv = Linv.T @ Linv
    return Ainv


@jit
def eval_grad_NLP(Y_error, Beta, G):
    return jnp.einsum('tk,kl,tli->i', Y_error, Beta, G)


# make sure that precision is positive definite (algorithm 3.3 in Numerical Optimization)
# def make_pos_def(A, Alpha, beta=1e-8):
#     # determine what Alpha needs to be in order for A + diag(Alpha) to be positive definite
#
#     # initial amount to add to matrix
#     if jnp.min(jnp.diag(A)) > 0:
#         tau = beta
#     else:
#         tau = beta - jnp.min(jnp.diag(A))
#
#     # use cholesky decomposition to check positive-definiteness of A
#     while jnp.isnan(jnp.linalg.cholesky(A)).any():
#         # increase precision of prior until posterior precision is positive definite
#         A += tau * jnp.diag(Alpha)
#
#         # increase prior precision
#         tau = max([2. * tau, beta])
#
#     return A, (1. + tau / 2.) * Alpha

# determine what Alpha needs to be in order for A = H + diag(Alpha) to be positive definite
def make_pos_def(A, Alpha, beta=1e-8):
    # make sure matrix is not already NaN
    assert not jnp.isnan(A).any(), "Matrix contains NaN, cannot make positive definite."

    # make sure alpha is not zero
    Alpha = jnp.clip(Alpha, beta, jnp.inf)

    # initial amount to add to matrix
    tau = 0.

    # use cholesky decomposition to check positive-definiteness of A
    while jnp.isnan(jnp.linalg.cholesky(A + tau*jnp.diag(Alpha))).any():
        # increase prior precision
        # print("adding regularization")
        tau = max([2. * tau, beta])

    # tau*jnp.diag(Alpha) is the amount that needed to be added for A to be P.D.
    # make alpha = (1 + tau)*alpha so that alpha + H is P.D.

    return A + tau*jnp.diag(Alpha), (1. + tau) * Alpha
