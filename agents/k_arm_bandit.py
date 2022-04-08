'''
  File: k_arm_bandit.py
 
  Author: Thomas Kost
  
  Date: 07 April 2022
  
  @breif k arm bandit agent
 '''

from builtins import Exception, dict
import numpy as np

class k_arm_bandit:
    def __init__(self, k, learning_config:dict)->None:
        '''
        k : number of arms
        learning_config : Dictionary with the following fields
            Mandatroy:
                Decisions: "e-greedy", "greedy"
                Tracking: "mean", "alpha_exp"
                Initialization: array of initial estimates length k

            Optional:
                Alpha: [0,1] for alpha exponential tracking
                eps: [0,1] for e-greedy choice

        '''
        self.k = k
        self.estimates = learning_config["Initialization"]
        self.N = np.zeros(k)
        self.decision = learning_config["Decisions"]
        self.tracking = learning_config["Tracking"]
        self.config = learning_config

    def choose(self):
        '''
        Pick an action based on current estimates
        '''
        # Determine Action
        if self.decision == "e-greedy":
            greedy_action = self.config["eps"] < np.random.rand()
            if greedy_action:
                return np.argmax(self.estimates)
            else:
                return int(np.random.rand()*self.k)

        elif self.decision == "greedy":
            return np.argmax(self.estimates)

        else:
            raise Exception("No such decision metric")

    def update_estimate(self, action_taken, Reward):
        '''
            action taken: result of calling k_arm_bandit.choose
            Reward: result of that action
        '''
        # Update Estimates
        if self.tracking == "mean":
            self.N[action_taken] += 1
            self.estimates[action_taken] += (Reward - self.estimates[action_taken])/self.N[action_taken]

        elif self.tracking == "alpha_exp":
            self.estimates[action_taken] += self.config["Alpha"]*(Reward - self.estimates[action_taken])

        else:
            raise Exception("No such Tracking metric")

