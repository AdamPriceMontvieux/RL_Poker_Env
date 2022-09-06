import gym
from gym import spaces
import numpy as np
from sqlalchemy import false
from card import Card


class VMEnv(gym.Env):

    def __init__(self, training_agent, other_agents):
        self.deck = np.arange(52)
        self.community_cards = np.zeros(52,5)
        self.hand_values = {0: 'High Card', 1: 'Pair', 2: 'Two Pair', 3: 'Three Of A Kind',
        4: 'Straight', 5: 'Flush', 6: 'Full House', 7: 'Four Of A Kind', 8: 'Stright Flush'}
        self.all_agents = [].append(training_agent)
        self.all_agents.extend(other_agents)
        self.hands = np.ones(6,52)
        self.chips = np.ones(6) * 100
        self.folded = np.zeros(6)
        self.round_bets = np.zeros(6)
        self.other_agents = other_agents
        self.training_agent = training_agent
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
        self.action_space = spaces.Discrete(3)

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
        self.round_step = 0
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
        self.training_agent_chips = self.chips[self.training_agent]

    def reset(self, seed=None, return_info=False, options=None):
        self.set_up()
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        self.training_agent_step(self.training_agent, action)
        return self.simulate_until_next_turn()
        
    def agent_step(self, agent, action):
        # Fold
        if action == 0:
           return self.fold()
        # Call
        elif action == 1:
            call_size = self.bet_size - self.round_bets[agent.ID]
            if call_size > self.chips[agent]:
                return self.fold(agent)
            else:
                self.chips[agent] -= call_size
                self.round_bets[agent.ID] += call_size
                self.pot_size += call_size
        # Raise
        else:
            bet_size = self.bet_size - self.round_bets[agent.ID] + 1
            if bet_size > self.chips[agent]:
                return self.fold(agent)
            else:
                self.chips[agent] -= bet_size
                self.round_bets[agent.ID] += bet_size
                self.pot_size += bet_size

    def fold(self, agent):
        self.folded[agent.ID] = True
        reward = -self.round_bets[agent] 
        done = True
        info = None
        return None, reward, done, info

    def simulate_until_next_turn(self):
        for a in self.other_agents:
            if self.is_betting_round_over():
                self.progress_game_step()

            self.agent_step(a, np.random.randint(0,3))

        return observation, reward, done, info

    def is_betting_round_over(self):
        mask = np.where(self.folded == 0)
        bets = self.round_bets[mask]
        if bets.min() == bets.max(): 
            return True
        return False

    def progress_game_step(self):
        self.bet_size = 1
        agents = [].append(self.training_agent)
        agents.extend(self.other_agents)
        self.round_bets = {a: 0 for a in agents}
        self.game_step += 1

    def showdown(self):
        

    def render(self, mode="ansi"):
        
        return ''

    def score_hand(hand):
        flush = hand.sum(axis=0).sum(axis=1)
        is_flush = np.max(flush) >= 5
        card_values = hand.sum(axis=0).sum(axis=0)
        count = 0
        hand_high = 8
        is_stright = False
        stright_high = 5
        for i, s in enumerate(card_values): 
            if s >= 1:
                count += 1
                hand_high = i
                if count >=5:
                    stright_high = i
                    is_stright = True
            else:
                count = 0
        if is_stright and is_flush:
            return 8, stright_high, hand_high    
        is_four_of_a_kind = np.isin(card_values, 4)
        if np.sum(is_four_of_a_kind):
            return 7, np.argwhere(is_four_of_a_kind==True)[0][0], hand_high
        is_three_of_a_kind = np.isin(card_values, 3)
        is_pair = np.isin(card_values, 2)
        if np.sum(is_pair) and np.sum(is_three_of_a_kind):
            return 6, np.argwhere(is_three_of_a_kind==True)[0][0], hand_high
        if is_flush:
            return 5, np.argwhere(hand.sum(axis=0)[np.argmax(flush)]==1)[-1][0], hand_high
        if is_stright:
           return 4, stright_high, hand_high
        if np.sum(is_three_of_a_kind):
            return 3, np.argwhere(is_three_of_a_kind==True)[0][0], hand_high
        if np.sum(is_pair) >= 2:
            return 2, np.argwhere(is_pair==True)[1][0], hand_high
        if np.sum(is_pair):
            return 1, np.argwhere(is_pair==True)[0][0], hand_high
        return 0, 0, hand_high
    
