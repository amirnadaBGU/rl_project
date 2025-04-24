import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent
import numpy as np
# Paths to your domain and instance files
base_path = './'  # Adjust the path accordingly
domain_file = base_path + 'casino_mdp.rddl'
instance_file = base_path + 'casino_instance.rddl'

def cast_action_to_true_bool(action_dict):
    return {key: np.bool(True) for key, value in action_dict.items()}

# Create the environment
myEnv = pyRDDLGym.make(domain=domain_file, instance=instance_file)

# Create a random agent
agent = RandomAgent(action_space=myEnv.action_space,
                    num_actions=myEnv.max_allowed_actions)

# Run the episode
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
myEnv.close()