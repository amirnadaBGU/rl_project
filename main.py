import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# Paths to your domain and instance files
base_path = './'  # Adjust the path accordingly
domain_file = base_path + 'casino_mdp.rddl'
instance_file = base_path + 'casino_instance.rddl'

def cast_action_to_true_bool(action_dict):
    return {key: np.bool_(True) for key, value in action_dict.items()}

def jackplociy():
    total_reward = 0
    state, _ = myEnv.reset()
    action = {'action2':True, 'action1':False}
    for step in range(myEnv.horizon):
        next_state, reward, done, info, _ = myEnv.step(action)
        total_reward += reward

        print(f'step       = {step}')
        print(f'state      = {state}')
        print(f'action     = {action}')
        print(f'next state = {next_state}')
        print(f'reward     = {reward}\n')
        state = next_state

        if action == {'action2':True, 'action1':False}:
            action == {'action1': True, 'action2': False}

        if done:
            print('---')
            break

    return total_reward
    print(f'Episode ended with total reward: {total_reward}')

def random_policy():
    # Create a random agent
    agent = RandomAgent(action_space=myEnv.action_space,
                        num_actions=myEnv.max_allowed_actions)

    total_reward = 0
    state, _ = myEnv.reset()
    for step in range(myEnv.horizon):
        action = agent.sample_action()
        action = cast_action_to_true_bool(action)
        next_state, reward, done, info, _ = myEnv.step(action)
        total_reward += reward

        print(f'step       = {step}')
        print(f'state      = {state}')
        print(f'action     = {action}')
        print(f'next state = {next_state}')
        print(f'reward     = {reward}\n')

        state = next_state
        if done:
            break

    print(f'Episode ended with total reward: {total_reward}')

def plot_reward_histogram(reward_list):
    # Bin edges every 0.05 units
    bin_width = 0.05
    bins = np.arange(min(reward_list), max(reward_list) + bin_width, bin_width)

    plt.hist(reward_list, bins=bins, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Reward Histogram, Expected Reward over {len(reward_list)} steps: {sum(reward_list)/len(reward_list)}')
    plt.tight_layout()
    plt.show()

# Create the environment
myEnv = pyRDDLGym.make(domain=domain_file, instance=instance_file)

# Run the episode
total_reward = 0
reward_vector = []
for k in range(10):
    reward_vector.append(jackplociy())

plot_reward_histogram(reward_vector)

myEnv.close()