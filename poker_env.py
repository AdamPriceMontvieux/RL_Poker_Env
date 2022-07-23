import gym
from gym import spaces
import numpy as np
from card import Card
from call import call
from fold import fold
from raise_bet import raise_bet


class VMEnv(gym.Env):

    def __init__(self, agents):
        self.deck = np.arange(52)
        self.community_cards = np.zeros(52,5)
        self.hand_values = {0: 'High Card', 1: 'Pair', 2: 'Two Pair', 3: 'Three Of A Kind',
        4: 'Straight', 5: 'Flush', 6: 'Full House', 7: 'Four Of A Kind', 8: 'Stright Flush'}
        self.hands = {a: '' for a in agents}
        self.chips = {a: 100 for a in agents}
        self.folded = {a: False for a in agents}
        self.agents = agents
        self.game_step = 0
        self.round_step = 0
        self.num_agents = len(self.agents)
        self.dealer = 0
        self.small_blind = 1
        self.big_blind = 2
        self.current_actor = 3
        self.game_number = -1
        self.pot_size = 0
        self.bet_size = 0
        self.raise_count = 0

    def _get_obs(self, agent):
        community_cards_state = np.sum(self.community_cards, axis=0)
        hand_state = np.sum(self.hands[agent], axis=0)
        return np.concatenate(community_cards_state, hand_state, self.chips[agent])

    def _get_info(self):
        return {}

    def set_up(self, seed=None, return_info=False, options=None):
        self.deck = np.arange(52)
        np.random.shuffle(self.deck)
        self.deck = self.deck.to_list()
        self.game_step = 0
        for a in self.agents:
            hand = np.zeros((2,52))
            hand[0,:] = Card(self.deck.pop()).vec
            hand[1,:] = Card(self.pop()).vec
            self.hands[a] = hand
        self.game_number += 1
        self.dealer = self.game_number % self.num_agents
        self.small_blind = (self.game_number+1) % self.num_agents
        self.big_blind = (self.game_number+2) % self.num_agents
        self.current_actor = (self.game_number+3) % self.num_agents
        self.pot_size = 0
        self.bet_size = 0
        self.raise_count = 0

    def reset(self, seed=None, return_info=False, options=None):

        self.set_up()

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):


        return observation, reward, done, info

    def render(self, mode="ansi"):

        return ''

    def score_hand(self, hand):
        all_cards = np.concatenate((self.community_cards, hand))
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
    
