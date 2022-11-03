import gym
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from gym import spaces
import mpu

class HeuristicPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        HandScores = mpu.io.read('filename.pickle')
        print(args[2]['difficulty'])

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
        obs = obs_batch[0][0:28] + obs_batch[0][29:]
        print(obs)
        # if value > rand * 1.5:
        #     return 2 #Raise
        # elif value > rand:
        #     return 1 #Call
        # else: 
        #     return 0 #Fold

        return [np.random.random_integers(0,3) for x in obs_batch], [], {}


