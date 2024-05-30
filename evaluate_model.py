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
from stable_baselines3.common.env_util import make_vec_env
from dotenv import load_dotenv
load_dotenv()
import argparse 


from integrator_rl import CanteraEnvII
from utils import compare_actions, simulation_test

parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str, default='time_reward')
parser.add_argument('--best_model', type=str, default='logs/20240523-074813/best_model.zip')
parser.add_argument('--mechanism_file', type=str, default='/Users/elotech/Documents/CODES/SCI-ML/integrator_rl/mechanism_files/ch4_53species.yaml')
parser.add_argument('--fuel', type=str, default='CH4')
parser.add_argument('--model', type=str, choices=['dqn', 'ppo', 'sac'], default='dqn')
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
num_points = 10

env = CanteraEnvII(mechanism_file, fuel, phi, T_initial, P_initial, species_to_track, residence_time, num_points, debug=args.debug)

print(env.action_space)

env_vec = make_vec_env(lambda: env, n_envs=1)
# load the best model
model =PPO("MlpPolicy", env_vec, verbose=1)
model = PPO.load(args.best_model, env=env_vec)

# compare the effects of different actions
actions = [0, 1, 2, 3]

_ = compare_actions(env, actions, model=model, experiment_name=args.exp)