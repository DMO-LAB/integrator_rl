import equinox.internal as eqxi

import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_solve, lu_factor
#from scipy.linalg import lu_factor, lu_solve
from functools import partial
from jax import jit

#jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_platform_name', 'cpu')

NEWTON_MAXITER = 4

def norm(x):
    """Compute RMS norm."""
    return jnp.linalg.norm(x) / x.size ** 0.5

def solve_linearized_system3(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):
    d = 0
    y = y_predict.copy()
    dy_norm_old = -1e10
    converged = False
    k = 0
    cond = True
    break_loop = False
    while break_loop == False and converged == False:
        #print(k)
        f = fun(t_new, y)

        if not np.all(np.isfinite(f)):
            #print("hereee2!")
            break

        dy = solve_lu(LU, c * f - psi - d)
        dy_norm = norm(dy / scale)
        #"hereee6!"
        if dy_norm_old < 0:
            rate = -100000.0
        else:
            rate = dy_norm / dy_norm_old

        if (dy_norm_old > -1000.0 and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
            #print("hereee3!")
            break_loop = True
            #break
        else:
            #print("misc:", c * f - psi - d)
            y += dy
            d += dy

        
            if (dy_norm == 0 or
                    dy_norm_old > -1000.0 and rate / (1 - rate) * dy_norm < tol):
                converged = True
                "hereee3!"
                #break

        

            dy_norm_old = dy_norm

            

            k += 1
        #cond = jnp.logical_and(~break_loop, ~converged)
        print("break loop is ", break_loop, "| converged is ", converged)
        print("not break loop is ", ~break_loop, "| not converged is ", ~converged)
        print("condition 1:", break_loop == False and converged == False)
        print("condition 2:", jnp.logical_and(~break_loop, ~converged))
        print("condition 2:", jnp.logical_and(break_loop == False, converged == False))
    return converged, k + 1, y, d


def fun1(args):
    y, d, dy, k, _ = args
    converged = False
    return y, d, k, converged

def fun2(args):
    y, d, dy, k, converged = args

    return y+dy, d+dy, k+1, converged



def solve_linearized_system2(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):

    
    y = y_predict.copy()
    d = jnp.zeros(y.shape)
    dy_norm_old = -1e10
    converged = False
    k = 0
    cond = True
    break_loop = False
    while break_loop == False and converged == False:
        #k, converged, break_loop, dy_norm_old, d, y, rate =  vars
        f = fun(t_new, y)

        dy = solve_lu(LU, c * f - psi - d)
        dy_norm = norm(dy / scale)
        
        #if dy_norm_old < 0:
        #    rate = -10000.0
        #else:
        rate = dy_norm / dy_norm_old

        #y += dy
        #d += dy

        cond_ = jnp.logical_or(dy_norm == 0, jnp.logical_and(dy_norm_old > -1.0, rate / (1 - rate) * dy_norm < tol)) #rate / (1 - rate) * dy_norm < tol #dy_norm == 0 or rate != -1.0 and rate / (1 - rate) * dy_norm < tol
        

        

        break_loop = k > NEWTON_MAXITER or (dy_norm_old > -1000.0 and (rate >= 1 or rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol))

        converged_temp = dy_norm == 0 or dy_norm_old > -1000.0 and rate / (1 - rate) * dy_norm < tol
        args = [y, d, dy, k, converged_temp]
        y, d, k, converged = jax.lax.cond(break_loop, fun1, fun2, args)

        # if break_loop:
        #     #print("hereee3!")
        #     #break_loop = True
        #     y = y
        #     d = d
        #     converged = False
        #     #break
        # else:
        #     y = y+dy
        #     d = d+dy

        #     converged = dy_norm == 0 or dy_norm_old > -1000.0 and rate / (1 - rate) * dy_norm < tol

        #     k += 1


        #print(break_loop, converged)
        #print(break_loop == False and converged == False)
            #print("dy_norm_old:", dy_norm, dy_norm_old)
            #k += 1

        dy_norm_old = dy_norm
            #print("dy_norm_old:", dy_norm, dy_norm_old)
        #cond = jnp.logical_and(break_loop == False, converged == False)

            #print("cond", cond)
            


        #return [k + 1, converged, break_loop, dy_norm_old, d, y, rate]

    return converged, k+1, y, d

#@partial(jit, static_argnums=(0,6))
def solve_linearized_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol, fun_args):
    def body_fun(vars):

        k, converged, break_loop, dy_norm_old, d, y, rate, max_iter_reached =  vars
        f = fun(t_new, y, fun_args)

        dy = solve_lu(LU, c * f - psi - d)
        dy_norm = norm(dy / scale)
        
        rate = dy_norm / dy_norm_old 

        break_loop = jnp.logical_and(dy_norm_old > -1000.0, jnp.logical_or(rate >= 1, rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol))
        #break_loop = k > NEWTON_MAXITER or (dy_norm_old > -1000.0 and (rate >= 1 or rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol))

        converged_temp = jnp.logical_or(dy_norm == 0, jnp.logical_and(dy_norm_old > -1000.0, rate / (1 - rate) * dy_norm < tol)) #rate / (1 - rate) * dy_norm < tol #dy_norm == 0 or rate != -1.0 and rate / (1 - rate) * dy_norm < tol
        #converged_temp = jnp.bool(jnp.where(cond, 1, 0))

        #converged_temp = dy_norm == 0 or dy_norm_old > -1000.0 and rate / (1 - rate) * dy_norm < tol
        args = [y, d, dy, k, converged_temp]
        y, d, k, converged = jax.lax.cond(break_loop, fun1, fun2, args)

        dy_norm_old = dy_norm

        return [k + 1, converged, break_loop, dy_norm_old, d, y, rate, max_iter_reached]

    def cond_fun(vars):
        k, converged, break_loop, dy_norm, d, y, rate, max_iter_reached =  vars

        cond = jnp.logical_and(break_loop == False, converged == False)
        max_iter_reached = k > NEWTON_MAXITER
        cond = jnp.logical_and(cond, max_iter_reached == False)
        return cond

    y = jnp.array(y_predict.copy())
    d = jnp.zeros(y.shape)
    dy_norm_old = jnp.array(-1000.0)
    rate = jnp.array(-1000.0)
    k = 0
    converged = jnp.bool(0)
    break_loop = jnp.bool(0)
    max_iter_reached = jnp.bool(0)
    init_val = [k, converged, break_loop, dy_norm_old, d, y, rate, max_iter_reached]
    k, converged, break_loop, dy_norm, d, y, rate, max_iter_reached = eqxi.while_loop(cond_fun, body_fun, max_steps = NEWTON_MAXITER, init_val = init_val, kind = "bounded", checkpoints=5)
    #k, converged, break_loop, dy_norm, d, y, rate, max_iter_reached = jax.lax.while_loop(cond_fun, body_fun, init_val)
    #print(converged, dy_norm)
    return converged, k, y, d


@partial(jit, static_argnums=(0,6))
def solve_linearized_system_old(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):
    def body_fun(vars):

        k, converged, break_loop, dy_norm_old, d, y, rate =  vars
        f = fun(t_new, y)

        dy = solve_lu(LU, c * f - psi - d)
        dy_norm = norm(dy / scale)
        
        rate = dy_norm / dy_norm_old 

        y += dy
        d += dy

        cond = jnp.logical_or(dy_norm == 0, jnp.logical_and(dy_norm_old != -1.0, rate / (1 - rate) * dy_norm < tol)) #rate / (1 - rate) * dy_norm < tol #dy_norm == 0 or rate != -1.0 and rate / (1 - rate) * dy_norm < tol
        converged = jnp.bool(jnp.where(cond, 1, 0))

        break_loop = jnp.logical_and(dy_norm_old != -1.0, jnp.logical_or(rate >= 1, rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol))

        dy_norm_old = dy_norm

        return [k + 1, converged, break_loop, dy_norm_old, d, y, rate]

    def cond_fun(vars):
        k, converged, break_loop, dy_norm, d, y, rate =  vars

        cond = jnp.logical_and(break_loop == False, converged == False)
        #cond = jnp.logical_and(cond, ~max_iter_reached)
        return cond

    y = jnp.array(y_predict.copy())
    d = jnp.zeros(y.shape)
    dy_norm_old = jnp.array(-1.0)
    rate = jnp.array(-1.0)
    k = 0
    converged = jnp.bool(0)
    break_loop = jnp.bool(0)
    #max_iter_reached = jnp.bool(0)
    init_val = [k, converged, break_loop, dy_norm_old, d, y, rate]
    k, converged, break_loop, dy_norm, d, y, rate = eqxi.while_loop(cond_fun, body_fun, max_steps = NEWTON_MAXITER, init_val = init_val, kind = "bounded")
    #print(converged, dy_norm)
    return converged, k, y, d
