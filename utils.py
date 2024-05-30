
from datetime import datetime
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, SAC


def simulation_test(env, action, render=False, safe_fig_path=None, model=None):
    # given an action, run the entire simulation and return the final state
    # this is used to test the environment
    obs, _ = env.reset()
    done = False
    while not done:
        if model is not None:
            action, _ = model.predict(obs)
        # check if action is not int then it would ne a model
        obs, rewards, done, truncated, info = env.step(action)
    print(f"Total time taken: {env.total_time} seconds")
    if render:
        env.render(plot_reference=True, save_path=safe_fig_path)
    return obs


def compare_actions(env, actions, model=None, experiment_name=None):
    # Compare the effects of different actions on the simulation
    # This is used to test the environment
    if experiment_name is None:
        experiment_name = "experiment"
    base_path = f"results/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(base_path, exist_ok=True)
    results = []
    if model is not None:
        actions_list = actions + [model]
    else:
        actions_list = actions
    for action in actions_list:
        try:
            if isinstance(action, PPO) or isinstance(action, SAC) or isinstance(action, DQN):
                action_name = "best_model"
                print(f"Running best model")
                save_fig_path = f"{base_path}/{action_name}.png"
                obs = simulation_test(env, action, render=False, safe_fig_path=save_fig_path, model=action)
                print(f"Time taken: {env.total_time} seconds, Error: {env.full_error}")
                action_result = {
                    "action":action_name,
                    "time_taken": np.log10(env.time_taken_list),    
                    "total_time":env.total_time,
                    "accuracy":env.accuracy_list,
                    "time_indices":env.time_indices,
                    "temperature":env.temperatures,
                    "action_list":env.action_list,
                    "full_error":env.full_error,
                }
                results.append(action_result)
                continue

            integrator = env.integrators[action // len(env.tolerances)]
            rtol = env.tolerances[action % len(env.tolerances)]
            action_name = f"{integrator}_{rtol}"
            print(f"Running action: {action_name}")
            save_fig_path = f"{base_path}/{action_name}.png"
            obs = simulation_test(env, action, render=False, safe_fig_path=save_fig_path)
            print(f"Time taken: {env.total_time} seconds, Error: {env.full_error}")
            action_result = {
                "action":action_name,
                "time_taken": np.log10(env.time_taken_list),
                "total_time":env.total_time,    
                "accuracy":env.accuracy_list,
                "time_indices":env.time_indices,
                "temperature":env.temperatures,
                "full_error":env.full_error,
            }
            results.append(action_result)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"Error: {e}")
            continue
    
    # save the results
    with open(f"{base_path}/results.pkl", 'wb') as f:
        pickle.dump(results, f)
    # plot the results
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    symbols = ['o', 's', 'D', '^', 'v', 'p', 'P']
    for i, result in enumerate(results):
        ax[0].plot(result["time_indices"], result["time_taken"], label=f"Log Time taken for {result['action']}", marker=symbols[i],markersize=5)
        ax[1].plot(result["time_indices"], result["accuracy"], label=f"Accuracy for {result['action']}", marker=symbols[i],markersize=5)
        ax[2].plot(result["time_indices"], result["temperature"], label=f"Temperature for {result['action']}", marker=symbols[i],markersize=5)
    ax[2].plot(np.linspace(0, env.simulation_time, env.num_points)*env.num_points/10, env.reference_solution[:, -1], label="Reference", linestyle='--')
    ax[0].set_xlabel("Time index")
    ax[0].set_ylabel("Log Time taken (s)")
    ax[0].set_title("Time taken for different actions")
    ax[0].legend()
    
    ax[1].set_xlabel("Time index")
    ax[1].set_ylabel("Negative Log Accuracy")
    ax[1].set_title("Negative Log Accuracy for different actions")
    ax[1].legend()
    
    ax[2].set_xlabel("Time index")
    ax[2].set_ylabel("Temperature (K)")
    ax[2].set_title("Temperature for different actions")
    ax[2].legend()
    
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/results.png")
    plt.show()
    plt.close()
    
    # plot a bar chart of the time taken for each action
    time_taken = [result["total_time"] for result in results]
    fig, ax = plt.subplots()
    ax.bar([result["action"] for result in results], time_taken)
    ax.set_xlabel("Action")
    ax.set_ylabel("Total Time taken (s)")
    ax.set_title("Total Time taken for each action")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{base_path}/time_taken.png")
    plt.show()
    
    # plot a bar chart of the error for each action
    error = [result["full_error"] for result in results]
    fig, ax = plt.subplots()
    ax.bar([result["action"] for result in results], error)
    ax.set_xlabel("Action")
    ax.set_ylabel("Error")
    ax.set_title("Error for each action")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{base_path}/error.png")
    plt.show()
    # plot a distribution of action for each action taken by model (action_list) if model is provided
    if model is not None:
        action_list = results[-1]["action_list"]
        action_distribution = [action_list.count(action) for action in actions]
        fig, ax = plt.subplots()
        ax.bar([f"Action {action}" for action in actions], action_distribution)
        ax.set_xlabel("Action")
        ax.set_ylabel("Frequency")
        ax.set_title("Action distribution")
        plt.tight_layout()
        plt.savefig(f"{base_path}/action_distribution.png")
        plt.show()
        
    # plot a line continous distribution of action for each action taken by model (action_list) if model is provided
    if model is not None:
        action_list = results[-1]["action_list"]
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0, env.simulation_time, env.num_points)*env.num_points/10, action_list, label="Action", marker='o')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Action")
        ax.set_title("Action distribution")
        plt.tight_layout()
        plt.savefig(f"{base_path}/action_distribution_line.png")
        plt.show()
    return results
    