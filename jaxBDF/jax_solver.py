import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jacfwd
from BDF import initialize_BDF_solver, solveODE
import time
from JaxChem import JaxChem
import argparse

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

class JaxChemSolver:
    def __init__(self, file_name, P, num_points, max_steps):
        self.jC = JaxChem(file_name=file_name)
        self.P = P
        self.num_points = num_points
        self.max_steps = max_steps
        self.eta = jnp.array(10.0)  # Ignore
        self.alpha = jnp.ones(shape=(self.jC.num_species,)) * 100  # Ignore

    def sigmoid(self, x, eta):
        x = jnp.minimum(10, x)
        x = jnp.maximum(-10, x)
        return 1 / (1 + jnp.exp(-eta * x))

    def compute_source_terms(self, state, alpha_eta):
        alpha = alpha_eta[0]
        eta = alpha_eta[1]
        alpha = self.sigmoid(alpha, eta)

        state = state[None, :]

        self.jC.T = state[:, 0:1].T
        self.jC.Patm = alpha_eta[2]

        self.jC.Y = state[:, 1:].T

        self.jC.update_thermo_properties()
        self.jC.compute_forward_rate_constants()
        self.jC.compute_reverse_rate_constants()
        self.jC.compute_net_production_rates(alpha)

        self.jC.species_rates = (self.jC.net_production_rates * self.jC.molecular_weights).T / self.jC.density
        partial_molar_cp = self.jC.cp_R * self.jC.Ru
        mixture_cp_mole = jnp.sum(partial_molar_cp * self.jC.X, axis=0, keepdims=True)
        mixture_cp_mass = mixture_cp_mole / self.jC.mix_molecular_weight
        partial_molar_enthalpies = self.jC.h_RT * self.jC.Ru * self.jC.T
        partial_mass_enthalpies = partial_molar_enthalpies / self.jC.molecular_weights.T

        self.jC.temperature_source = -jnp.sum(self.jC.species_rates * partial_mass_enthalpies / mixture_cp_mass, axis=0)

        source = jnp.concatenate((self.jC.temperature_source[:, None], self.jC.species_rates.T), axis=1)[0, :]

        return source

    def rhs(self, t, y, params):
        y = jnp.array(y)
        return self.compute_source_terms(y, params)

    def solve(self, Y0, t_end, phi):
        t0 = 0.0  # Initial time
        ts = jnp.linspace(t0, t_end * .998, self.num_points)  # Initialize ts (make it end slightly before t_end)

        tspan = [t0, t_end]
        solver_state = initialize_BDF_solver(fun=self.rhs, t_span=tspan, y0=Y0, rtol=1e-8, atol=1e-8, fun_args=(self.alpha, self.eta, self.P))

        start = time.time()
        ts, ys, num_steps = solveODE((self.alpha, self.eta, self.P), solver_state, self.max_steps, self.rhs, ys, ts, Y0)
        end = time.time()
        print(f"Took {end - start} seconds.")

        return ts, ys

    def plot_results(self, ts, ys):
        plt.plot(ts[:], ys[:, 0], "-o")
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Profile')
        plt.show()

if __name__ == '__main__':
# Example usage
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file_name", type=str, default="gri30.cti")
    argparser.add_argument("--fuel", type=str, default="CH4")
    args = argparser.parse_args()
    file_name = args.file_name
    
    Tinit = 950.0
    P = 1
    num_points = 1000
    max_steps = 1000
    phi = 0.7
    t_end = 0.0001

    jC = JaxChem(file_name=file_name)
    Y0 = jC.init_from_equi_ratio(oxidizer={'O2': 1.0, 'N2': 3.76}, fuel={args.fuel: 1.0}, phi=phi)
    Y0 = jnp.maximum(1e-16, Y0)  # Give the mass fractions a small value
    Y0 = jnp.hstack([jnp.array([Tinit]), Y0])

    solver = JaxChemSolver(
        file_name=file_name,
        P=P,
        num_points=num_points,
        max_steps=max_steps
    )
    ts, ys = solver.solve(Y0, t_end, phi)
    solver.plot_results(ts, ys)
