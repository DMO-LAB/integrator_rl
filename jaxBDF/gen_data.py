import jax
import jax.numpy as jnp
import equinox.internal as eqxi
from JaxChem import JaxChem
import os
#import matplotlib.pyplot as plt
from jax import jacfwd
from BDF import initialize_BDF_solver, solveODE
import time
import os
#import mpi4jax
#from mpi4py import MPI
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()

jC = JaxChem(file_name = "/Users/elotech/Documents/CODES/SCI-ML/files2Elo/ch4_53species.yaml")

def sigmoid(x, eta):
    x = jnp.minimum(10,x)
    x = jnp.maximum(-10, x)
    return 1/(1 + jnp.exp(-eta*x))

def compute_source_terms(state, alpha_eta):

    alpha = alpha_eta[0]
    eta = alpha_eta[1]
    alpha = sigmoid(alpha, eta)

    state = state[None, :]
    
    jC.T = state[:,0:1].T
    jC.Patm = alpha_eta[2]
    
    jC.Y = state[:,1:].T

    jC.update_thermo_properties()
    jC.compute_forward_rate_constants()
    jC.compute_reverse_rate_constants()
    jC.compute_net_production_rates(alpha)

    jC.species_rates = (jC.net_production_rates * jC.molecular_weights).T/jC.density
    partial_molar_cp = jC.cp_R * jC.Ru
    mixture_cp_mole = jnp.sum(partial_molar_cp * jC.X, axis = 0, keepdims=True)
    mixture_cp_mass = mixture_cp_mole/jC.mix_molecular_weight
    partial_molar_enthalpies = jC.h_RT * jC.Ru * jC.T
    partial_mass_enthalpies = partial_molar_enthalpies/jC.molecular_weights.T
    
    jC.temperature_source = -jnp.sum(jC.species_rates * partial_mass_enthalpies/mixture_cp_mass, axis = 0)
    
    source = jnp.concatenate((jC.temperature_source[:,None], jC.species_rates.T), axis = 1)[0,:]

    return source

def rhs(t, y, params):
    y = jnp.array(y)

    return compute_source_terms(y, params)



start = time.time()
Tinit = 950.0 #initial T
P = 1 #pressure in atm
num_points = 1000 #number of points to save

ys = jnp.zeros((num_points, jC.num_species+1)) # empty array to hold solution

max_steps = 1000 #maximum number of steps solver can take
eta = jnp.array(10.0) #ignore
alpha = jnp.ones(shape = (jC.num_species,))*100 #ignore

Y0 = jC.init_from_equi_ratio(oxidizer = {'O2': 1.0, 'N2': 3.76}, fuel = {"CH4":1.0}, phi = 0.7) #initialize Y
Y0 = jnp.maximum(1e-16, Y0) #give the mass fractions a small value
T = jnp.array([Tinit]) 
Y0 = jnp.hstack([T,Y0])

t0 = 0.0 #initial time
t1 = 5 # end time of simulation in seconds
ts = jnp.linspace(t0,t1*.998,num_points) #initialize ts (make it end slightly before t1)

tspan = [t0, 5.0]
solver_state = initialize_BDF_solver(fun = rhs, t_span = tspan, y0 = Y0, rtol=1e-8, atol=1e-8, fun_args=(alpha, eta, 1.0))

start = time.time()
ts, ys, num_steps = solveODE((alpha, eta, P), solver_state, max_steps, rhs, ys, ts, Y0)
end = time.time()
print(f"Took {end - start} seconds.")



plt.plot(ts[:], ys[:,0], "-o")
plt.show()