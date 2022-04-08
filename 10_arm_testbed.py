'''
  File: 10_arm_testbed.py
 
  Author: Thomas Kost
  
  Date: 07 April 2022
  
  @breif simulate 10 arm bandit with different estimations
 '''

from statistics import mean
import numpy as np
from agents.k_arm_bandit import k_arm_bandit
from environments.slots import slots
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
experiments = 200
steps = 10000
k=10
drift = 0.01
mean = {
    "Decisions": "e-greedy",
    "Tracking": "mean",
    "Initialization": np.zeros(k),
    "eps": 0.1
}

exp = {
    "Decisions": "e-greedy",
    "Tracking": "alpha_exp",
    "Initialization": np.zeros(k),
    "eps": 0.1,
    "Alpha":0.1
}

# Declare bandits and slots
mean_bandit  = k_arm_bandit(k, mean)
exp_bandit   = k_arm_bandit(k, exp)
mean_bandit_drift  = k_arm_bandit(k, mean)
exp_bandit_drift   = k_arm_bandit(k, exp)

mean_actions = np.zeros((steps, experiments))
exp_actions  = np.zeros((steps, experiments))
mean_actions_drift = np.zeros((steps, experiments))
exp_actions_drift  = np.zeros((steps, experiments))

mean_rewards = np.zeros((steps, experiments))
exp_rewards  = np.zeros((steps, experiments))
mean_rewards_drift = np.zeros((steps, experiments))
exp_rewards_drift  = np.zeros((steps, experiments))
static_opts = np.zeros(experiments)
drift_opts = np.zeros(experiments)

for j in tqdm(range(experiments)):
    temp_mean_actions = np.zeros(steps)
    temp_exp_actions  = np.zeros(steps)
    temp_mean_actions_drift = np.zeros(steps)
    temp_exp_actions_drift  = np.zeros(steps)

    temp_mean_rewards = np.zeros(steps)
    temp_exp_rewards  = np.zeros(steps)
    temp_mean_rewards_drift = np.zeros(steps)
    temp_exp_rewards_drift  = np.zeros(steps)

    static_slots = slots(k, 0, 1, 3)
    drift_slots  = slots(k,drift, 1, 3)

    static_opts[j] = static_slots.optimal_choice
    drift_opts[j] =  drift_slots.optimal_choice

    for i in range(steps):
    
        # No drift
        mean_action = mean_bandit.choose()
        mean_reward = static_slots.return_reward(mean_action)
        mean_bandit.update_estimate(mean_action, mean_reward)
        temp_mean_actions[i] = mean_action
        temp_mean_rewards[i] = mean_reward

        exp_action = exp_bandit.choose()
        exp_reward = static_slots.return_reward(exp_action)
        exp_bandit.update_estimate(exp_action, exp_reward)
        temp_exp_actions[i] = exp_action
        temp_exp_rewards[i] = exp_reward

        # Drift
        mean_action_d = mean_bandit_drift.choose()
        mean_reward_d = static_slots.return_reward(mean_action_d)
        mean_bandit_drift.update_estimate(mean_action_d, mean_reward_d)
        temp_mean_actions_drift[i] = mean_action_d
        temp_mean_rewards_drift[i] = mean_reward_d

        exp_action_d = exp_bandit_drift.choose()
        exp_reward_d = static_slots.return_reward(exp_action_d)
        exp_bandit_drift.update_estimate(exp_action_d, exp_reward_d)
        temp_exp_actions_drift[i] = exp_action_d
        temp_exp_rewards_drift[i] = exp_reward_d
        
        # Cause drift
        drift_slots.vary_dists()
        
    mean_actions[:,j] = temp_mean_actions
    exp_actions[:,j] = temp_exp_actions
    mean_actions_drift[:,j] = temp_mean_actions_drift
    exp_actions_drift[:,j] = temp_exp_actions_drift    

    mean_rewards[:,j] = temp_mean_rewards
    exp_rewards[:,j] = temp_exp_rewards
    mean_rewards_drift[:,j] = temp_mean_rewards_drift
    exp_rewards_drift[:,j] = temp_exp_rewards_drift

# Analysis
mean_avg_reward = np.mean(mean_rewards, axis=1)
exp_avg_reward = np.mean(exp_rewards, axis=1)
mean_avg_reward_drift = np.mean(mean_rewards_drift, axis=1)
exp_avg_reward_drift = np.mean(exp_rewards_drift, axis=1)

avgt_opt_mean_actions = np.sum(mean_actions ==static_opts, axis=1)/experiments
avgt_opt_exp_actions  = np.sum(exp_actions ==static_opts, axis=1)/experiments
avgt_opt_mean_actions_drift = np.sum(mean_actions_drift ==drift_opts, axis=1)/experiments
avgt_opt_exp_actions_drift  = np.sum(exp_actions_drift ==drift_opts, axis=1)/experiments
n = np.arange(steps)

# Create Plots
plt.subplot(121)
plt.plot( n, avgt_opt_mean_actions, label = "mean, no drift")
plt.plot( n, avgt_opt_exp_actions, label = "exp, no drift")
plt.plot( n, avgt_opt_mean_actions_drift, label = "mean, drift")
plt.plot( n, avgt_opt_exp_actions_drift, label = "exp, drift")
plt.title("% Opt")
plt.legend()
plt.subplot(122)
plt.plot(n, mean_avg_reward, label = "mean, no drift")
plt.plot( n, exp_avg_reward, label = "exp, no drift")
plt.plot( n, mean_avg_reward_drift, label = "mean, drift")
plt.plot( n, exp_avg_reward_drift, label = "exp, drift")
plt.title("Reward")
plt.legend()
plt.show()