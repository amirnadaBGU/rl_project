import multiprocessing as mp
from pyRDDLGym.core.policy import RandomAgent
import pyRDDLGym
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

DOMAIN = 'casino_mdp.rddl'
INSTANCE = 'casino_instance.rddl'

# worker initializer
def _init_worker():
    global ENV          # כל תהליך יקבל עותק פרטי
    ENV = pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE)

# -------------------------------------------------
# 2.  פרק יחיד (מסתמך על ENV הגלובלי בתהליך)
# -------------------------------------------------
def run_episode(_seed):
    total = 0.0
    state, _ = ENV.reset(seed=_seed)      # reset קובע זריעת אקראיות שונה
    action = {'action1': False, 'action2': True}

    for _ in range(ENV.horizon):
        state, r, done, *_ = ENV.step(action)
        total += r
        # החלפה בין שתי הפעולות
        action = {'action1': not action['action1'],
                  'action2': not action['action2']}
        if done:
            break
    return total          # ייאסף לתהליך הראשי

def cast_action_to_true_bool(action_dict):
    return {key: np.bool_(True) for key, value in action_dict.items()}

def run_episode_random(_seed):
    agent = RandomAgent(action_space=ENV.action_space,
                        num_actions=ENV.max_allowed_actions)
    total = 0.0
    state, _ = ENV.reset(seed=_seed)      # reset קובע זריעת אקראיות שונה

    for _ in range(ENV.horizon):
        action = agent.sample_action()
        action = cast_action_to_true_bool(action)
        state, r, done, *_ = ENV.step(action)
        total += r
        if done:
            break
    return total          # ייאסף לתהליך הראשי

def optimal_policy(_seed):
    def get_optimal_action(state,action):
        if state['S'] == 0:
            action['action1'] = False
            action['action2'] = True
        elif state['S'] == 1:
            action['action1'] = True
            action['action2'] = False
        elif state['S'] == 2:
            number = random.randint([0,1])
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
    state, _ = ENV.reset(seed=_seed)
    action = {'action1':False, 'action2':False}
    for step in range(ENV.horizon):
        action = get_optimal_action(state,action)
        next_state, reward, done, info, _ = ENV.step(action)
        total_reward += reward
        if action['action1'] == True and action['action2'] == True:
            print('Two actions in parallel')
            break
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

# -------------------------------------------------
# 3.  בלוק ההפעלה המקביל
# -------------------------------------------------
if __name__ == "__main__":
    N_EPISODES = 1_000_000
    N_PROCS    = mp.cpu_count()           # מספר ליבות פיזיות / לוגיות

    with mp.Pool(processes=N_PROCS,
                 initializer=_init_worker) as pool:

        # pool.map => מחלק את טווח הפרקים בין התהליכים
        rewards = pool.map(optimal_policy, range(N_EPISODES),
                           chunksize=1000)   # שולח באצ'ים של 1000 לקריאה אחת

    print(f'ממוצע תגמול: {np.mean(rewards):.4f}')

    plot_reward_histogram(rewards)

