import jax
import jax.numpy as jnp
import equinox.internal as eqxi
from jax.scipy.linalg import lu_solve, lu_factor
from solve_BDF_system import solve_linearized_system2, solve_linearized_system
import equinox.internal as eqxi
from functools import partial
from jax import jit
from BDF_utils import _update_D, norm
from jax import jacfwd

#jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_platform_name', 'cpu')

jnp.set_printoptions(precision=16)

MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10
LARGE_NUM = 1e5

def exitfun1(arg_vars):
    error_const, order, D, scale, h_abs, A, B, n_equal_steps, error_norm, safety = arg_vars
    return [order, h_abs, D, n_equal_steps]

def exitfun2(arg_vars):
    error_const, order, D, scale, h_abs, A, B, n_equal_steps, error_norm, safety = arg_vars


    arg_vars = [error_const, order, D, scale, -1, 0]
    error_m_norm = jax.lax.cond(order > 1, pmnorm_fun1, pmnorm_fun2, arg_vars)

    arg_vars = [error_const, order, D, scale, 1, 2]
    error_p_norm = jax.lax.cond(order < MAX_ORDER, pmnorm_fun1, pmnorm_fun2, arg_vars)  

    error_norms = jnp.array([error_m_norm, error_norm, error_p_norm])


    factors = error_norms ** (-1 / (jnp.arange(3) + order))

    delta_order = jnp.argmax(factors) - 1
    
    order += delta_order

    factor = jnp.minimum(MAX_FACTOR, safety * jnp.max(factors))

    h_abs *= factor

    D = _update_D(D, order, factor, A, B)
    n_equal_steps = 0

    return [order, h_abs, D, n_equal_steps]

def J_LU_update(args):
    t_new, y_predict, c, _, _, I, fun_args = args["vars"]
    fun = args["fun"]
    #t_new, y_predict, c, _, _, I, jac = args
    #J = jac(t_new, y_predict, fun_args)
    J = jacfwd(fun, argnums=1)(0.0, y_predict, fun_args)
    LU = lu_factor(I - c * J)
    current_jac = True
    return J, LU, current_jac


def no_J_LU_update(args):
    _, _, _, J, LU, _, fun_args = args["vars"]
    return J, LU, False


def newton_solver_update(args):
    t_new, y_predict, c, psi, LU, scale, converged, n_iter, y_new, d, newton_tol, fun_args = args["vars"]
    fun = args["fun"]
    converged, n_iter, y_new, d = solve_linearized_system(fun, t_new, y_predict, c, psi, LU, 
                                                          lu_solve, scale, newton_tol, fun_args)
    return converged, n_iter, y_new, d

def no_newton_solver_update(args):
    t_new, y_predict, c, psi, LU, scale, converged, n_iter, y_new, d, newton_tol, fun_args = args["vars"]

    return converged, n_iter, y_new, d
    
    #self.LU = None


def update_h_D(args):
    h_abs, order, A, B, D, n_iter, atol, rtol, y_new, d, error_const, scale, n_equal_steps = args

    factor = 0.5
    h_abs *= factor
    D = _update_D(D, order, factor, A, B)
    n_equal_steps = 0

    #give large negative value for safety and large pos value for error_norm: should not affects results downstream
    step_accepted = False
    return factor, h_abs, D, n_equal_steps, -1000.0, scale, 100000.0, step_accepted

def no_h_D_update(args):
    h_abs, order, A, B, D, n_iter, atol, rtol, y_new, d, error_const, scale, n_equal_steps = args

    safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

    scale = atol + rtol * jnp.abs(y_new)
    error = error_const[order] * d
    error_norm = norm(error / scale)


    h_abs, D, n_equal_steps, step_accepted = funx(h_abs, D, safety, error_norm, order, A, B, n_equal_steps)
    
    # return zero for factor: just means its not updated: I don't think it's used as is downstream
    # so should not change anything
    return 1000000.0, h_abs, D, n_equal_steps, safety, scale, error_norm, step_accepted

def last_fun_true(vars):
    t_bound, D, order, t, h_abs, A, B, t_new, n_equal_steps = vars
    t_new = t_bound
    D = _update_D(D, order, jnp.abs(t_new - t) / h_abs, A, B)
    n_equal_steps = 0

    return t_new, D, n_equal_steps

def last_fun_false(vars):
    t_bound, D, order, t, h_abs, A, B, t_new, n_equal_steps = vars

    return t_new, D, n_equal_steps

def pmnorm_fun1(vars):
    error_const, order, D, scale, i, j =  vars
    error_m = error_const[order + i] * D[order + j]
    error_m_norm = norm(error_m / scale)

    return error_m_norm

def pmnorm_fun2(vars):
    error_const, order, D, scale, i, j =  vars

    return LARGE_NUM

# def pnorm_fun1(vars):
#     error_const, order, D, scale, i, i =  vars
#     error_p = error_const[order + 1] * D[order + 2]
#     error_p_norm = norm(error_p / scale)
#     return error_m_norm

# def pnorm_fun2(vars):
#     error_const, order, D, scale =  vars

#     return LARGE_NUM

def fun1(args):

    h_abs, D, safety, error_norm, order, A, B, n_equal_steps =  args
    factor = jnp.maximum(MIN_FACTOR, safety * error_norm ** (-1 / (order + 1)))
    h_abs *= factor
    D_new = _update_D(D, order, factor, A, B)
    n_equal_steps = 0
    step_accepted = False
    #print(D_new)
    return h_abs, D_new, n_equal_steps, step_accepted


def fun2(args):
    h_abs, D, safety, error_norm, order, A, B, n_equal_steps =  args
    #print(h_abs)
    step_accepted = True
    return h_abs, D, n_equal_steps, step_accepted
    
#@partial(jit, static_argnums=(0,1))
#@jax.jit
def funx(h_abs, D, safety, error_norm, order, A, B, n_equal_steps):
    args = h_abs, D, safety, error_norm, order, A, B, n_equal_steps
    h_abs, D, n_equal_steps, step_accepted = jax.lax.cond(error_norm > 1, fun1, fun2, args)
    
    return h_abs, D, n_equal_steps, step_accepted