import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.integrate import solve_ivp
import gymnasium as gym
import cantera as ct
import math
import numpy as np
from scipy.integrate import solve_ivp
import datetime as dt
from stable_baselines3 import DQN, PPO, SAC
from datetime import datetime
from stable_baselines3.common import logger
import os
import pickle
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
load_dotenv()
from stable_baselines3.common.logger import configure


class CanteraEnvII(gym.Env):
    def __init__(self, mechanism_file, fuel, phi, T_initial, P_initial, species_to_track, simulation_time, num_points, neptune_logger=None, **kwargs):
        super(CanteraEnvII, self).__init__()
        
        self.mechanism_file = mechanism_file
        self.fuel = fuel
        self.phi = phi
        self.T_initial = T_initial
        self.P_initial = P_initial
        self.species_to_track = species_to_track
        self.simulation_time = simulation_time
        self.num_points = num_points
        self.integrators = ['BDF']
        self.tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
        self.y = []
        self.debug = kwargs.get('debug', False)
        self.action_space = gym.spaces.Discrete(len(self.integrators) * len(self.tolerances))
        self.neptune_logger = neptune_logger
        #self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(species_to_track) + 1,), dtype=np.float64)
        
        self.gas = ct.Solution(mechanism_file)
        self.reference_solution = self.calculate_reference_solution()
        # reference solution is 1000, interpolate the values to match the number of points
        self.reference_solution = self.interpolate_reference_solution(num_points)
        self.reset()

    def step(self, action):
        integrator = self.integrators[action // len(self.tolerances)]
        rtol = self.tolerances[action % len(self.tolerances)]
        atol = rtol # check
        self.action_list.append(action)
        #self.atol = min(1e-4, rtol)
        if self.debug:
            print(f"Time index: {self.time_index}, Integrator: {integrator}, RTOL: {rtol}, ATOL: {atol}")
        # Perform the simulation
        try:
            y, T, P = self.solve_with_scipy(self.gas, integrator, 0, self.dt_cfd, rtol, atol)
            self.gas.Y = y
            self.gas.TP = T, P
        except ct.CanteraError as ce:
            print(f"Warning: CanteraError at time {self.time_index} with integrator: {integrator} and RTOL: {rtol}")
            if self.debug:
                print(f"Error: {ce}")
            reward = -50
            self.total_reward += reward
            self.average_reward = (self.average_reward * self.time_index + reward) / (self.time_index + 1)
            done = True
            self.full_error = self.compare_final_simulation(self.time_index)
            print(f"Simulation Stopped at time index: {self.time_index} with average reward: {self.average_reward} and total reward: {self.total_reward}, total time taken: {self.total_time} seconds \n")
            return self._get_state(), reward, done, True, {}
        
        self.time_indices.append(self.time_index)
        # Track temperature and species profiles
        self.temperatures.append(self.gas.T)
        for spec in self.species_to_track:
            self.species_profiles[spec].append(self.gas[spec].X[0])
        
        # Calculate reward based on time and accuracy
        current_state = self._get_state()
        reference_state = self.reference_solution[self.time_index]
        
        reward = self.cal_reward(current_state, reference_state)
        self.time_index += 1
        done = self.time_index >= self.num_points
        if done:
            reward += self.end_of_episode_reward()
            self.full_error = self.compare_final_simulation()
            print(f"Completed! Time index: {self.time_index} with average reward: {self.average_reward} and total reward: {self.total_reward}, total time taken: {self.total_time} seconds \n")
            print(f"****************************************************************************************************\n")
        truncated = False

        return current_state, reward, done, truncated, {"integrator": integrator, "rtol": rtol}
    
    def compare_final_simulation(self, time_step=None):
        # combine the temperature and species profiles
        if time_step:
            full_simulation = np.vstack([self.temperatures[0:time_step], [self.species_profiles[spec][0:time_step] for spec in self.species_to_track]])
            reference_simulation = self.reference_solution[0:time_step].T
        else:
            full_simulation = np.vstack([self.temperatures, [self.species_profiles[spec] for spec in self.species_to_track]])
            reference_simulation = self.reference_solution.T
        # calculate the error
        error = np.linalg.norm(full_simulation - reference_simulation)
    
        return error
    
    def interpolate_reference_solution(self, num_points):
        new_times = np.linspace(0, self.simulation_time, num_points)
        interpolated_solution = np.empty((num_points, self.reference_solution.shape[1]))

        for i in range(self.reference_solution.shape[1]):
            interpolated_solution[:, i] = np.interp(new_times, self.reference_times, self.reference_solution[:, i])
        return interpolated_solution

    def reward_function(self, time, k=1):
        return np.log10(time) / k

    def cal_reward(self, current_state, reference_state):
        error = self.cal_error(current_state, reference_state)
        if error < 0:
            error_reward = error - 5
        else:
            error_reward = error
        self.accuracy_list.append(error)
        time_reward = -(self.reward_function(self.time_taken, k=10) + self.time_taken * 10)
        reward = time_reward #+ error_reward
        self.total_reward += reward
        self.average_reward = self.total_reward / (self.time_index + 1)
        if self.debug:
            print(f"Time index: {self.time_index}, Error: {error}, Error Reward: {error_reward}, Time Reward: {time_reward}, Total Reward: {reward}, Time taken: {self.time_taken} seconds")
        return reward
    
    def cal_error(self, current_state, reference_state):
        # get the normalized error between the current state and the reference state
        error = np.linalg.norm(current_state - reference_state)
        error = -np.log10(error)
        return error
    
    def end_of_episode_reward(self):
        return -np.log10(self.total_time) * 10
    
    def reset(self, seed=None):
        self.gas.set_equivalence_ratio(self.phi, self.fuel, {'O2': 1, 'N2': 3.76})
        self.gas.TP = self.T_initial, self.P_initial
        self.time_index = 0
        self.total_time = 0
        self.dt_cfd = self.simulation_time / self.num_points
        self.t_end = self.simulation_time
        self.atol = 1e-12
        self.temperatures = []
        self.action_list = []
        self.time_indices = []
        self.time_taken_list = []
        self.accuracy_list = []
        self.average_reward = 0
        self.total_reward = 0   
        self.species_profiles = {spec: [] for spec in self.species_to_track}
        return self._get_state(), {}
    
    def _get_state(self):
        state = np.hstack([self.gas[spec].X[0] for spec in self.species_to_track] + [self.gas.T])
        return state
    
    def solve_with_scipy(self, gas, method, time, dt_cfd, rtol=1e-6, atol=1e-15):
        y0 = np.hstack((gas.Y, gas.T))

        def derivatives(t, y):
            gas.Y = y[:-1]
            gas.TP = y[-1], gas.P
            dydt = np.zeros_like(y)
            r = gas.net_production_rates
            rho = gas.density
            dydt[:gas.n_species] = r * gas.molecular_weights / rho
            dydt[-1] = - np.dot(gas.partial_molar_enthalpies, r) / (rho * gas.cp_mass)
            return dydt
        
        start_time = dt.datetime.now()
        solution = solve_ivp(derivatives, [time, time + dt_cfd], y0, method=method, rtol=rtol, atol=atol)
        end_time = dt.datetime.now()
        self.time_taken_list.append((end_time - start_time).total_seconds())
        self.time_taken = (end_time - start_time).total_seconds()
        self.total_time += self.time_taken  
        # if self.debug:
        #     print(f"Time index: {self.time_index}, Time taken: {self.time_taken} seconds with integrator: {method} and RTOL: {rtol}")
        return solution.y[:-1, -1], solution.y[-1, -1], gas.P

    def calculate_reference_solution(self):
        gas = ct.Solution(self.mechanism_file)
        oxidizer = {'O2': 1, 'N2': 3.76}
        gas.set_equivalence_ratio(self.phi, self.fuel, oxidizer)
        gas.TP = self.T_initial,ct.one_atm

        reactor = ct.ConstPressureReactor(gas)
        network = ct.ReactorNet([reactor])

        times = np.linspace(0, self.simulation_time, 1000)
        reference_solution = []

        for time in times:
            network.advance(time)
            state = np.hstack([reactor.thermo[spec].X[0] for spec in self.species_to_track] + [reactor.T])
            reference_solution.append(state)
        self.reference_times = times
        return np.array(reference_solution)

    def render(self, mode='human', **kwargs):
        plot_reference = kwargs.get('plot_reference', False)
        save_path = kwargs.get('save_path', None)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        times = np.linspace(0, self.simulation_time, self.num_points)
        
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
        
        # plot the accuracy and time taken
        ax3.plot(self.time_indices, self.accuracy_list)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel("Negative Log Error")
        ax3.set_title('Negative Log Error Profile')
        
        ax4.plot(self.time_indices, self.time_taken_list)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Log Time Taken (s)')
        ax4.set_title('Log Time Taken Profile')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Create and wrap the environment
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mechanism_file', type=str, default='/Users/elotech/Documents/CODES/SCI-ML/integrator_rl/mechanism_files/ch4_53species.yaml')
    parser.add_argument('--fuel', type=str, default='CH4')
    parser.add_argument('--model', type=str, choices=['dqn', 'ppo', 'sac'], default='dqn')
    parser.add_argument('--neptune_logger', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    
    args = parser.parse_args()
    
    chosen_model = args.model
    mechanism_file = args.mechanism_file
    fuel = args.fuel
    print(f"Using mechanism file: {mechanism_file} and fuel: {fuel}")

    phi = 1.0
    T_initial = 900
    P_initial = ct.one_atm
    species_to_track = ['CH4', 'O2', 'CO2', 'H2O']
    residence_time = 10
    num_points = 50

    env = CanteraEnvII(mechanism_file, fuel, phi, T_initial, P_initial, species_to_track, residence_time, num_points, debug=args.debug)

    env_vec = make_vec_env(lambda: env, n_envs=1)
    log_path = f"./logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
    # Define the checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=log_path, name_prefix='rl_model')
    eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path, eval_freq=1000, deterministic=True, render=False)
        
    # Choose the model
    if chosen_model == 'dqn':
        model = DQN("MlpPolicy", env_vec, verbose=1)
    elif chosen_model == 'ppo':
        model = PPO("MlpPolicy", env_vec, verbose=1)
    else:
        model = SAC("MlpPolicy", env_vec, verbose=1)
    # Set up logger
    tmp_path = "/tmp/integrator_rl/logs"
    new_logger = configure(tmp_path, ["csv", "stdout"])
    model.set_logger(new_logger)
    # Train the model
    model.learn(total_timesteps=20000, callback=[checkpoint_callback, eval_callback])

    # Save the model
    model.save("rl_model")
    env.close()
