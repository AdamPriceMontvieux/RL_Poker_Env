import gym
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from gym import spaces
import mpu

class HeuristicPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if args[2]['difficulty'] == 0:
            self.HandScores = mpu.io.read('hand_value_table_norm_int8_10.pickle')
        if args[2]['difficulty'] == 1:
            self.HandScores = mpu.io.read('hand_value_table_norm_int8_100.pickle')
        else: 
            self.HandScores = mpu.io.read('hand_value_table_norm_int8_1000.pickle')

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
        obs = obs_batch[0][0:28] + obs_batch[0][29:]
        obs = np.array(np.where(obs==1)[0], dtype=np.int8)
        print(obs)
        value = self.HandScores[obs.tobytes()]
        rand = np.random.random()
        if value > rand * 1.5:
            return [2], [], {} #Raise
        elif value > rand:
            return [1], [], {} #Call
        else: 
            return [0], [], {} #Fold



