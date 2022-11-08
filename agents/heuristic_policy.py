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
            self.HandScores = mpu.io.read('Heurisitc_Val_Tables/hand_value_table_norm_int8_24_10.pickle')
        if args[2]['difficulty'] == 1:
            self.HandScores = mpu.io.read('Heurisitc_Val_Tables/hand_value_table_norm_int8_24_100.pickle')
        else: 
            self.HandScores = mpu.io.read('Heurisitc_Val_Tables/hand_value_table_norm_int8_24_1000.pickle')
        #self.observation_space = spaces.Dict({
        #        "hand": spaces.Box(0, 1, shape=(24, )),
        #        "community": spaces.Box(0, 1, shape=(24, ))
        #    })
        #self.observation_space_struct = get_base_struct_from_space(self.observation_space)
        #self.action_space = spaces.Discrete(3)

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
        obs = obs_batch[0][0:24] + obs_batch[0][24:]
        obs = np.array(np.where(obs==1)[0], dtype=np.int8)
        value = self.HandScores[obs.tobytes()]
        rand = np.random.random()
        if value > rand:
            return [2], [], {} #Raise
        elif value > (rand/4):
            return [1], [], {} #Call
        else: 
            return [0], [], {} #Fold

    def get_weights(self):
        return None

    def set_weights(self, weights):
        return None


