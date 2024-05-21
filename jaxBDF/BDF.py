import jax
import jax.numpy as jnp
import equinox.internal as eqxi
from jax.scipy.linalg import lu_solve, lu_factor
from solve_BDF_system import solve_linearized_system
import equinox.internal as eqxi
from functools import partial
from jax import jit
from BDF_utils import compute_R, _update_D, norm, validate_first_step, select_initial_step, check_y
from BDF_functions import no_J_LU_update, J_LU_update, newton_solver_update, no_newton_solver_update, funx
from BDF_functions import update_h_D, no_h_D_update, last_fun_true, last_fun_false, pmnorm_fun1, pmnorm_fun2, exitfun1, exitfun2
from jax import jacfwd, jacrev
from JaxChem import JaxChem
import os
import collections
import equinox as eqx
import functools as ft


#jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_platform_name', 'cpu')



MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10
LARGE_NUM = 1e5




def initialize_BDF_solver(fun,  t_span, y0, t0 = 0.0, max_step=1000, first_step =  None,
                rtol=1e-3, atol=1e-6, jac_fun=None, fun_args =  None):
    
    
    states = {}
    states["max_step"] = max_step
    states["rtol"] = rtol
    states["atol"] = atol
    states["t"] = t_span[0]
    #states["fun"] =  fun
    states["y"] = check_y(y0)
    states["t_bound"] = t_span[-1]
    states["fun_args"] = fun_args
    states["direction"] = jnp.sign(states["t_bound"] - t0) if states["t_bound"] != t0 else 1
    #self.njev = 0
    #self.nlu = 0
    states["n"] = states["y"].size

    f = fun(states["t"], states["y"], fun_args)

    if first_step is None:
        states["h_abs"] = select_initial_step(fun, states["t"], states["y"], f,states["direction"], 1,
                                            rtol, atol, fun_args)
    else:
        states["h_abs"] = validate_first_step(first_step, states["t"], states["t_bound"])

    #self.h_abs_old = -1.0
    #self.error_norm_old = -1.0

    states["newton_tol"] = jnp.maximum(10 * jnp.finfo(float).eps / rtol, jnp.minimum(0.03, rtol ** 0.5))
    states["A"] = jnp.arange(0, MAX_ORDER + 1).reshape(-1, 1)
    states["B"] = jnp.arange(0, MAX_ORDER + 1)

    #print(self.A.shape, MAX_ORDER)
    if jac_fun is not None:
        states["J"] = _validate_jac(jac_fun, states["t"], states["y"], fun_args, states["n"])
    else:
        states["J"] = jnp.zeros((states["n"], states["n"]))
    #states["jac_fun"] = jac_fun
    states["I"] = jnp.identity(states["n"])
    

    kappa = jnp.array([0, -0.1850, -1/9, -0.0823, -0.0415, 0])
    states["gamma"] = jnp.hstack((0, jnp.cumsum(1 / jnp.arange(1, MAX_ORDER + 1))))
    states["alpha"] = (1 - kappa) * states["gamma"]
    
    states["error_const"] = kappa * states["gamma"] + 1 / jnp.arange(1, MAX_ORDER + 2)

    

    D = jnp.zeros((MAX_ORDER + 1, states["n"]))
    D = D.at[0].set(states["y"])
    D = D.at[1].set(f * states["h_abs"] * states["direction"])
    states["Dindex"] = jnp.arange(D.shape[0]) #needed to update psi and y_predict
    states["zerosD"] = jnp.zeros_like(D) #needed to update psi and y_predict
    states["D"] = D

    order = 1
    states["order"] = 1
    states["n_equal_steps"] = 0
    c = states["h_abs"] * states["alpha"][order]
    states["LU"] = lu_factor(states["I"] - c * states["J"])
    
    BDFInternalStates = ["t", "max_step", "atol","rtol","y", "t_bound",
                         "fun_args","direction","newton_tol","n","h_abs","alpha",
                         "error_const","J","A","B","I","LU",
                         "gamma", "Dindex",  "n_equal_steps", 
                         "order", "D",  "zerosD"]
    BDFState = collections.namedtuple("BDFState", BDFInternalStates)
    states_named_tuple = BDFState(*[states[k] for k in BDFInternalStates])
    return states_named_tuple

def _validate_jac(jac, t, y, fun_args, n):
    J = jac(t, y, fun_args)

    J = jnp.asarray(J)

    if J.shape != (n, n):
        raise ValueError("`jac` is expected to have shape {}, but "
                        "actually has {}."
                        .format((n, n), J.shape))
    
    return J

#@partial(jit, static_argnums=(2,3))
def advance_solution(fun_args, state, fun):

    A = state.A
    B = state.B

    t = state.t
    D = state.D
    h_abs = state.h_abs
    #max_step = self.max_step
    min_step = 0.0
    atol = state.atol
    rtol = state.rtol
    order = state.order
    alpha = state.alpha
    gamma = state.gamma
    error_const = state.error_const
    J = state.J

    #LU =  None

    step_accepted = False
    current_jac = False
    converged = False

    def body_fun(vars):
        t_new, y_new, h_abs, D, d, n_equal_steps, safety, scale, error_norm, step_accepted, J, LU, current_jac, converged =  vars

        h = h_abs * state.direction
        t_new = t + h

        vars = [state.t_bound, D, order, t, h_abs, A, B, t_new, n_equal_steps]
        llast_step = state.direction * (t_new - state.t_bound) > 0
        t_new, D, n_equal_steps = jax.lax.cond(llast_step, last_fun_true, last_fun_false, vars)

        h = t_new - t
        h_abs = jnp.abs(h)

        D_temp = jnp.where(state.Dindex[:,None] > order, state.zerosD, D)
        y_predict = jnp.sum(D_temp, axis=0)
        
        scale = atol + rtol * jnp.abs(y_predict)
        
        where_cond = jnp.logical_or(state.Dindex[:,None] > order,  state.Dindex[:,None] < 1)
        D_temp = jnp.where(where_cond, state.zerosD, D)
        psi = jnp.dot(D_temp.T, gamma) / alpha[order]

        
        c = h / alpha[order]

        LU = lu_factor(state.I - c * J)

        ###################################################################################################
        converged, n_iter, y_new, d = solve_linearized_system(fun, t_new, y_predict, c, psi, LU, lu_solve,
                                                                scale, state.newton_tol, fun_args)

        ###################################################################################################
        lax_cond = jnp.logical_and(converged == False, current_jac == False)
        #lax_cond = jnp.logical_and(~converged, ~current_jac)
        #args = {"vars": [t_new, y_predict, c, J, LU, state.I, fun_args], "fun": jax.tree_util.Partial(jac_fun)}
        args = {"vars": [t_new, y_predict, c, J, LU, state.I, fun_args], "fun": fun}
        J, LU, current_jac = jax.lax.cond(lax_cond, J_LU_update, no_J_LU_update, args)
        ###################################################################################################


        vars = [t_new, y_predict, c, psi, LU, scale, converged, n_iter, y_new, d, state.newton_tol, fun_args]
        args = {"vars": vars, "fun": jax.tree_util.Partial(fun)}

        converged, n_iter, y_new, d = jax.lax.cond(lax_cond, newton_solver_update, no_newton_solver_update, args)
        ###################################################################################################
        
        vars = [h_abs, order, A, B, D, n_iter, atol, rtol, y_new, d, error_const, scale, n_equal_steps]
        factor, h_abs, D, n_equal_steps, safety, scale, error_norm, step_accepted = jax.lax.cond(converged == False, update_h_D, no_h_D_update, vars)

        # ##########################################################################################################

        return [t_new, y_new, h_abs, D, d, n_equal_steps, safety, scale, error_norm, step_accepted, J, LU, current_jac, converged] 

    def cond_fun(vars):
        t_new, y_new, h_abs, D, d, n_equal_steps, safety, scale, error_norm, step_accepted, J, LU, current_jac, converged  =  vars


        return step_accepted == False

    d = jnp.zeros(state.y.shape)
    scale = jnp.zeros(state.y.shape)
    error_norm = 0.0
    safety = 0.0
    LU = lu_factor(state.I - 0.001 * J)
    init_state = [t, state.y, h_abs, D, d, state.n_equal_steps, safety, scale, error_norm, step_accepted, J, LU, current_jac, converged ]

    final_state = eqxi.while_loop(cond_fun, body_fun, max_steps = 10, init_val = init_state, kind = "checkpointed", checkpoints=1)
    #final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

    t_new, y_new, h_abs, D, d, n_equal_steps, safety, scale, error_norm, step_accepted, J, LU, current_jac, converged = final_state     

    n_equal_steps += 1

    state = state._replace(n_equal_steps=n_equal_steps, t=t_new, y=y_new, h_abs=h_abs, J=J, LU = LU)

    D = D.at[order + 2].set(d - D[order + 1])
    D = D.at[order + 1].set(d)
    
    D_temp = D.copy()
    D_temp2 = D.copy()

    #where_cond = jnp.logical_or(state.Dindex[:,None] > order+1,  state.Dindex[:,None] < 1)
    D_temp = jnp.where(state.Dindex[:,None] > order+1, state.zerosD, D)
    #D_temp = D_temp.at[order+2:].set(0.0)
    D = jnp.flip(jnp.cumsum(jnp.flip(D_temp, axis = 0), axis = 0), axis = 0)
    D = jnp.where(state.Dindex[:,None] > order, D_temp2, D) #D.at[order+1:].set(D_temp2[order+1:])

    arg_vars = [error_const, order, D, scale, h_abs, A, B, state.n_equal_steps, error_norm, safety]

    order, h_abs, D, n_equal_steps = jax.lax.cond(state.n_equal_steps < order + 1, exitfun1, exitfun2, arg_vars)
    
    state = state._replace(order=order, h_abs=h_abs, D=D, n_equal_steps=n_equal_steps)
    print(state.h_abs)
    return state, True

#@partial(jit, static_argnums=(2, 5))
#@ft.partial(jax.jit, static_argnums=5)
#@ft.partial(jax.jit, static_argnums=(4))
@jax.jit
def main_loop(fun, params, solver_state, ts, ys,max_steps):
    i = 1
    max_steps = 100000
    tend = solver_state.t_bound #tspan[-1]
    initial_state = [solver_state, ts, i, ys]

    def cond_fun(state):
        solver_state, t_save, i, _ = state
        return jnp.logical_and(i < len(ts), solver_state.t < tend)


    
    def body_fun(state):
        solver_state, ts, i, ys = state
        solver_state,_ = advance_solution(params, solver_state, fun)
        index = jnp.searchsorted(ts, solver_state.t)
        index = index.astype(
            "int" + ts.dtype.name[-2:]
        )  # Coerce index to correct type

        def cond_fun1(vars):
            j, _ = vars
            return j < index
        def body_fun1(vars):
            j, ys = vars
            t = ts[j]
            ys = ys.at[jnp.index_exp[j, :]].set(_bdf_interpolate(solver_state, t))
            return [j+1, ys]
        
        _, ys = eqxi.while_loop(cond_fun1, body_fun1, max_steps = ys.shape[0], init_val = [i, ys], kind = "bounded") #jax.lax.while_loop(cond_fun, body_fun, initial_state)
        #i = index
        return [solver_state, ts, index, ys]

    solver_state, ts, num_steps, ys = eqxi.while_loop(cond_fun, body_fun, max_steps = max_steps, init_val = initial_state, kind = "checkpointed", checkpoints=100) #jax.lax.while_loop(cond_fun, body_fun, initial_state)
    #solver_state, ts, num_steps, ys = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    return ts, ys, num_steps
    #return 

def solveODE(params, solver_state, max_steps, fun, ys, ts, Y0):
    solver_state = solver_state._replace(y=Y0)
    ys = ys.at[0].set(solver_state.y)
    fun_wrapped = jax.tree_util.Partial(fun)
    partial_func = ft.partial(main_loop, fun_wrapped)
    return partial_func(params=params, solver_state=solver_state, ts=ts, ys=ys, max_steps=max_steps)


@jax.jit
def _bdf_interpolate(state, t_eval):
    """
    interpolate solution at time values t* where t-h < t* < t

    definition of the interpolating polynomial can be found on page 7 of [1]
    """
    order = state.order
    t = state.t
    h = state.h_abs*state.direction
    D = state.D
    j = 0
    time_factor = 1.0
    order_summation = D[0]
    while_state = [j, time_factor, order_summation]

    def while_cond(while_state):
        j, _, _ = while_state
        return j < order

    def while_body(while_state):
        j, time_factor, order_summation = while_state
        time_factor *= (t_eval - (t - h * j)) / (h * (1 + j))
        order_summation += D[j + 1] * time_factor
        j += 1
        return [j, time_factor, order_summation]

    j, time_factor, order_summation = eqxi.while_loop(
        while_cond, while_body, while_state, max_steps = 100, kind = "bounded"
    )
    return order_summation


# Ts = [950.0, 1000.0, 1050.0, 1110.0, 1150.0, 1200.0]
# phis = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
# Ps = [1, 10, 20, 30]

# num_series = len(Ts)*len(phis)*len(Ps)
# for ii in range(num_series):
#     #print(ii%(len(Ts)*len(Ps)), int(ii/(len(Ts))))
#     print(ii%(len(Ts)), int(ii/(len(Ts)))%len(phis), int(ii/(len(Ts)*len(phis))))