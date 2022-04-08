'''
  File: slots.py
 
  Author: Thomas Kost
  
  Date: 07 April 2022
  
  @breif slot machine for k armed bandit
 '''
import numpy as np

class slots:

    def __init__(self, k, drift, std, max_mean)->None:
        '''
        k: number of slots
        drift
        '''
        self.k =k
        self.drift = drift
        self.stds = np.full(k, std,'float64')
        self.means = np.random.uniform(-max_mean, max_mean, k)
        self.optimal_choice = np.argmax(self.means)

    def return_reward(self, action):
        return np.random.normal(self.means[action], self.stds[action])

    def vary_dists(self):
        self.means += np.random.normal(0,self.drift,self.k)