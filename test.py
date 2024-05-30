import time
import matplotlib.pyplot as plt
from scipy.integrate import ode
import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
 
#gas = ct.Solution('ethane_mech.yaml')
gas = ct.Solution('gri30.yaml')
 
#gas.set_equivalence_ratio(0.7, 'nc7h16', 'O2:1.0, N2:3.76')
 
#gas.set_equivalence_ratio(1.2, 'C2H6', 'O2:1.0, N2:3.76')
gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1.0, N2:3.76')
 
TT = 900. # in Kelvin
 
gas.TP = TT, ct.one_atm
 
dt = 1.e-4
N = int(429*1e-6/dt)
#N = int(N/10)
# total time = 2e-5
N = 1000
 
 
def f(t, Y):
    T = Y[0] #np.exp(Y[0])
    YY = Y[1:] #np.exp(Y[1:])
    #YY = np.maximum(YY, 1e-16)
    gas.TPY = T, ct.one_atm, YY
    species_rates = gas.net_production_rates*gas.molecular_weights/gas.density
    species_h = gas.partial_molar_enthalpies/gas.molecular_weights
    temp_rate = -np.sum(species_rates*species_h/gas.cp_mass)
    return np.concatenate((np.array([temp_rate]),species_rates), axis = 0)
 
 
Y0 = np.concatenate((np.asarray(gas.TPY[0])[None], gas.Y), axis = 0)
 
Y0 = np.maximum(1e-12, Y0)
 
sol = solve_ivp(f, [0, 1.0], Y0, method='BDF', rtol=1e-6, atol=1e-15, dense_output = True) #, first_step = 1e-3)
plt.plot(sol.t, sol.y[0,:],"-o")
end = time.time()