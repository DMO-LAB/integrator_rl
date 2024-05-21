import jax
import jax.numpy as jnp
import equinox.internal as eqxi
from jax.scipy.linalg import lu_solve, lu_factor
from solve_BDF_system import solve_linearized_system, solve_linearized_system2
import equinox.internal as eqxi
from functools import partial
from jax import jit

#jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_platform_name', 'cpu')

jnp.set_printoptions(precision=16)

MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10
LARGE_NUM = 1e5

#@jax.jit
def rhs(t, y):

    ydot1 = -0.04*y[0] + 1e4 * y[1] * y[2]
    ydot2 = 0.04*y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1]**2
    ydot3 = 3e7 * y[1]**2

    return jnp.stack([ydot1, ydot2, ydot3])
#@jax.jit


#@jax.jit
def compute_R(factor, A, B):
    """Compute the matrix for changing the differences array."""
    M = jnp.zeros((MAX_ORDER + 1, MAX_ORDER + 1))
    M = M.at[jnp.index_exp[1:, 1:]].set((A[1:] - 1 - factor * B[1:]) / A[1:])
    M = M.at[jnp.index_exp[0]].set(1)
    R = jnp.cumprod(M, axis=0)

    return R

#@jax.jit
def _update_D(D, order, factor, A, B):

    U = compute_R(1, A, B)
    RU = compute_R(factor, A, B).dot(U)

    # only update order+1, order+1 entries of D
    # print(A.shape, B.shape, RU.shape)
    RU = jnp.where(jnp.logical_and(A <= order, B <= order), RU, jnp.identity(MAX_ORDER + 1))
    
    #print(RU.shape, D.shape)
    D = jnp.dot(RU.T, D)
    
    return D

def norm(x):
    """Compute RMS norm."""
    return jnp.linalg.norm(x) / x.size ** 0.5

def validate_first_step(first_step, t0, t_bound):
    """Assert that first_step is valid and return it."""
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")
    if first_step > jnp.abs(t_bound - t0):
        raise ValueError("`first_step` exceeds bounds.")
    return first_step

def select_initial_step(fun, t0, y0, f0, direction, order, rtol, atol, fun_args):

    if y0.size == 0:
        return jnp.inf

    scale = atol + jnp.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1, fun_args)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = jnp.maximum(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / jnp.maximum(d1, d2)) ** (1 / (order + 1))

    return jnp.minimum(100 * h0, h1)

def check_y(y0):
    """Helper function for checking arguments common to all solvers."""
    y0 = jnp.asarray(y0)
    if jnp.issubdtype(y0.dtype, jnp.complexfloating):
        raise ValueError("`y0` is complex, but the chosen solver does "
                             "not support integration in a complex domain.")

    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")

    if not jnp.isfinite(y0).all():
        raise ValueError("All components of the initial state `y0` must be finite.")


    return y0