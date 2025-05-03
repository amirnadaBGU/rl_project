import random

import pyRDDLGym
from dask.array.ma import average
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

def jack_plociy():
    total_reward = 0
    state, _ = myEnv.reset()
    action = {'action2':True, 'action1':False}
    for step in range(myEnv.horizon):
        next_state, reward, done, info, _ = myEnv.step(action)
        total_reward += reward
        if action['action1']==True and action['action2']==True:
            print('Two actions in parallel')
            break
        elif action['action2'] == True:
            action['action2']= False
            action['action1'] = True
        elif action['action1'] == True:
            action['action1'] = False
            action['action2'] = True
        if done:
            print('---')
            break

    return total_reward

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
        if done:
            break
    return total_reward

def optimal_policy():
    def get_optimal_action(state,action):
        if state['S'] == 0:
            action['action1'] = False
            action['action2'] = True
        elif state['S'] == 1:
            action['action1'] = True
            action['action2'] = False
        elif state['S'] == 2:
            number = random.randint(0,1)
            if number == 0:
                action['action1'] = True
                action['action2'] = False
            elif number == 1:
                action['action1'] = False
                action['action2'] = True
            else:
                print('error')
        return action

    total_reward = 0
    state, _ = myEnv.reset()
    action = {'action1':False, 'action2':False}
    for step in range(myEnv.horizon):
        action = get_optimal_action(state,action)
        action = get_optimal_action(state,action)
        next_state, reward, done, info, _ = myEnv.step(action)
        state = next_state
        total_reward += reward
        if done:
            print('---')
            break

    return total_reward

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

def plot_reward_convergence(xs,r2,r3,r4):
    plt.plot(xs, r2, label='Policy 2')  # first curve
    plt.plot(xs, r3, label='Policy 3')  # second curve
    plt.plot(xs, r4, label='Policy 4')  # third curve

    plt.xlabel('Number of simulations (N)')
    plt.ylabel('Average total reward after N trials')
    plt.title('Convergence of average reward vs. # simulations')
    plt.legend()
    plt.grid(True)
    plt.show()



# Create the environment
myEnv = pyRDDLGym.make(domain=domain_file, instance=instance_file)

# Run the episode
total_reward = 0
reward_vector_2 = []
reward_vector_3 = []
reward_vector_4 = []
xs = list(range(1, 1002, 10))
for k in xs:
    print(k)
    reward_2=0
    reward_3=0
    reward_4=0
    for i in range(k):
        reward_2 += jack_plociy()
        reward_3 += random_policy()
        reward_4 += optimal_policy()
    reward_vector_2.append(reward_2/k)
    reward_vector_3.append(reward_3 / k)
    reward_vector_4.append(reward_4 / k)

plot_reward_convergence(xs,reward_vector_2,reward_vector_3,reward_vector_4)

myEnv.close()