import gym
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from gym import spaces

class FolderPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.observation_space = dict({
        #    "state": spaces.Discrete(24)
        #    "obs": spaces.Discrete(29),
        #})

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
        return [0 for x in obs_batch], [], {}

    def get_weights(self):
        return None

    def set_weights(self, weights):
        return None