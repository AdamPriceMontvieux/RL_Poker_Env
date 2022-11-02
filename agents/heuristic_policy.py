import gym
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from gym import spaces

class HeuristicPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        matrix = np.load('../AITables/2_card_values_10000_norm.npy')

    def get_initial_state(self):
        return [random.choice([0, 1, 2])]

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):

        # values = matrix[obs_batch]

        
        # if value > rand * 1.5:
        #     return 2 #Raise
        # elif value > rand:
        #     return 1 #Call
        # else: 
        #     return 0 #Fold

        return 0, [], {}


