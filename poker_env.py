import gym
from gym import spaces
import numpy as np
import subprocess

# import the client class
from vm_gym.gRPC.client import Client
import time


class VMEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, agents):
        self.deck = np.arange(52)
        self.board = np.zeros(52,5)
        self.hand_values = {0: 'High Card', 1: 'Pair', 2: 'Two Pair', 3: 'Three Of A Kind',
        4: 'Straight', 5: 'Flush', 6: 'Full House', 7: 'Four Of A Kind', 8: 'Stright Flush'}
        self.hands = [[] for a in agents]
        self.agents = agents

    def _get_obs(self):
        # get the list of remaining files

        return obs

    def _get_info(self):
        return {}


    def set_up(self, seed=None, return_info=False, options=None):
        self.deck = np.arange(52)
        np.random.shuffle(self.deck)
        self.deck = self.deck.to_list()
    

    def reset(self, seed=None, return_info=False, options=None):

        self.set_up()

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        # Map the action (element of {0,1}) to the command to execute


        return observation, reward, done, info

    def render(self, mode="ansi"):

        return ''

    def score_hand(self, hand):
        all_cards = np.concatenate((self.board, hand))
        all_cards = all_cards.reshape(7,4,13)
        flush = all_cards.sum(axis=0).sum(axis=1)
        is_flush = np.max(flush) >= 5
        stright = all_cards.sum(axis=0).sum(axis=0)
        count = 0
        high = 0
        is_stright = False
        for i, s in enumerate(stright): 
            if s >= 1:
                count += 1
                high = i
                if count >=5:
                    is_stright = True
            else:
                count = 0

        if is_stright and is_flush:
            return 8, high
        if np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 4)):
            return 7, high
        if np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 2)) and np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 3)):
            return 6, high
        if is_flush:
            return 5, high
        if is_stright:
           return 4, high
        if np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 3)):
            return 3, high
        if np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 2)) >= 2:
            return 2, high
        if np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 2)):
            return 1, high
        return 0, high
    
