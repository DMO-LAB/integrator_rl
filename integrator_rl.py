import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.integrate import solve_ivp
import gymnasium as gym
import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp
import datetime as dt
from stable_baselines3 import DQN
# import dummy env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

class CanteraEnvII(gym.Env):
    def __init__(self, mechanism_file, fuel, phi, T_initial, P_initial, species_to_track, residence_time, num_points):
        super(CanteraEnvII, self).__init__()
        
        self.mechanism_file = mechanism_file
        self.fuel = fuel
        self.phi = phi
        self.T_initial = T_initial
        self.P_initial = P_initial
        self.species_to_track = species_to_track
        self.residence_time = residence_time
        self.num_points = num_points
        
        self.integrators = ['BDF', 'LSODA', 'RK45', 'RK23', 'Radau', 'DOP853']
        self.tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        
        self.action_space = gym.spaces.Discrete(len(self.integrators) * len(self.tolerances))
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(species_to_track) + 1,), dtype=np.float64)
        
        self.gas = ct.Solution(mechanism_file)
        self.reference_solution = self.calculate_reference_solution()
        self.reset()

    def step(self, action):
        integrator = self.integrators[action // len(self.tolerances)]
        rtol = self.tolerances[action % len(self.tolerances)]
        print(f"Time index: {self.time_index}, Integrator: {integrator}, RTOL: {rtol}")
        
        # Perform the simulation
        try:
            y, h, P = self.solve_with_scipy(self.gas, integrator, 0, self.dt_cfd, rtol, self.atol)
            self.gas.Y = y
            self.gas.HP = h, P
        except ct.CanteraError:
            print(f"Warning: CanteraError at time {self.time_index}")
            reward = -100
            done = True
            return self._get_state(), reward, done, True, {}
        
        # Track temperature and species profiles
        self.temperatures.append(self.gas.T)
        for spec in self.species_to_track:
            self.species_profiles[spec].append(self.gas[spec].X[0])
        
        # Calculate reward based on time and accuracy
        current_state = self._get_state()
        reference_state = self.reference_solution[self.time_index]
        accuracy = -np.linalg.norm(current_state - reference_state)
        
        reward = -self.time_taken + accuracy
        self.time_index += 1
        done = self.time_index >= self.num_points
        truncated = False
        
        return current_state, reward, done, truncated, {"integrator": integrator, "rtol": rtol}
    
    def reset(self, seed=None):
        self.gas.set_equivalence_ratio(self.phi, self.fuel, {'O2': 1, 'N2': 3.76})
        self.gas.TP = self.T_initial, self.P_initial
        self.time_index = 0
        self.dt_cfd = self.residence_time / self.num_points
        self.t_end = self.residence_time
        self.atol = 1e-15
        self.temperatures = []
        self.species_profiles = {spec: [] for spec in self.species_to_track}
        return self._get_state(), {}
    
    def _get_state(self):
        state = np.hstack([self.gas[spec].X[0] for spec in self.species_to_track] + [self.gas.T])
        return state
    
    def solve_with_scipy(self, gas, method, time, dt_cfd, rtol=1e-6, atol=1e-15):
        y0 = np.hstack((gas.Y, gas.enthalpy_mass))
        
        def derivatives(t, y):
            gas.Y = y[:-1]
            gas.HP = y[-1], gas.P
            dydt = np.zeros_like(y)
            r = gas.net_production_rates
            rho = gas.density
            dydt[:gas.n_species] = r * gas.molecular_weights / rho
            dydt[-1] = np.dot(gas.partial_molar_enthalpies, r)
            return dydt
        
        start_time = dt.datetime.now()
        solution = solve_ivp(derivatives, [time, time + dt_cfd], y0, method=method, rtol=rtol, atol=atol)
        end_time = dt.datetime.now()
        
        self.time_taken = (end_time - start_time).total_seconds()
        print(f"Time taken: {self.time_taken} seconds for {method} with RTOL: {rtol}")
        return solution.y[:-1, -1], solution.y[-1, -1], gas.P

    def calculate_reference_solution(self):
        gas = ct.Solution(self.mechanism_file)
        oxidizer = {'O2': 1, 'N2': 3.76}
        gas.set_equivalence_ratio(self.phi, self.fuel, oxidizer)
        gas.TP = self.T_initial, self.P_initial

        reactor = ct.IdealGasReactor(gas)
        network = ct.ReactorNet([reactor])

        times = np.linspace(0, self.residence_time, self.num_points)
        reference_solution = []

        for time in times:
            network.advance(time)
            state = np.hstack([reactor.thermo[spec].X[0] for spec in self.species_to_track] + [reactor.T])
            reference_solution.append(state)
        
        return np.array(reference_solution)

    def render(self, mode='human', **kwargs):
        plot_reference = kwargs.get('plot_reference', False)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
        
        times = np.linspace(0, self.residence_time, self.num_points)
        
        ax1.plot(times[:len(self.temperatures)], self.temperatures, label='Simulation')
        if plot_reference:
            ax1.plot(times[:len(self.reference_solution[:self.time_index])], self.reference_solution[:self.time_index, -1], label='Reference', linestyle='--')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Profile')

        for spec in self.species_to_track:
            ax2.plot(times[:len(self.species_profiles[spec])], self.species_profiles[spec], label=f"{spec} Simulation")
            if plot_reference:
                ax2.plot(times[:len(self.reference_solution[:self.time_index])], self.reference_solution[:self.time_index, self.species_to_track.index(spec)], label=f"{spec} Reference", linestyle='--')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Mass Fraction')
        ax2.set_title('Species Mass Fraction Profiles')
        ax2.legend()

        plt.tight_layout()
        plt.show()

def simulation_test(env, action, render=False):
    # given an action, run the entire simulation and return the final state
    # this is used to test the environment
    env.reset()
    done = False
    while not done:
        obs, rewards, done, truncated, info = env.step(action)
    if render:
        env.render(plot_reference=True)
    return obs

# Create and wrap the environment
if __name__ == "__main__":
    mechanism_file = '/Users/elotech/Documents/CODES/SCI-ML/mechanism_files/ethane_mech.yaml'
    fuel = 'C2H6'
    phi = 1.0
    T_initial = 1800
    P_initial = ct.one_atm
    species_to_track = ['C2H6', 'O2', 'CO2', 'H2O']
    residence_time = 0.0001
    num_points = 1000

    env = CanteraEnvII(mechanism_file, fuel, phi, T_initial, P_initial, species_to_track, residence_time, num_points)

    env_vec = make_vec_env(lambda: env, n_envs=1)
    # Define the checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='models/', name_prefix='dqn_cantera')

    # Define and train the RL agent
    model = DQN('MlpPolicy', env_vec, verbose=1)
    model.learn(total_timesteps=10000, callback=checkpoint_callback, progress_bar=True)
    
    # save the model
    model.save("dqn_cantera")

