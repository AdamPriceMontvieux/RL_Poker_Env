import gym
from gym import spaces
import numpy as np
#from sqlalchemy import false
from card import Card


class PokerEnv(gym.Env):

    def __init__(self, training_agent, other_agents):
        self.deck = np.arange(52)
        self.community_cards = np.zeros((5, 52))
        self.hand_values = {0: 'High Card', 1: 'Pair', 2: 'Two Pair', 3: 'Three Of A Kind',
        4: 'Straight', 5: 'Flush', 6: 'Full House', 7: 'Four Of A Kind', 8: 'Stright Flush'}
        self.all_agents = [training_agent]
        self.all_agents.extend(other_agents)
        self.other_agents = other_agents
        self.training_agent = training_agent
        self.game_step = 0
        self.round_step = 0
        self.num_agents = len(self.all_agents)
        self.dealer = 0
        self.small_blind = 1
        self.big_blind = 2
        self.current_actor = 3
        self.game_number = -1
        self.pot_size = 0
        self.bet_size = 0
        self.raise_count = 0
        self.action_space = spaces.Discrete(3)

    def get_obs(self, agent):
        community_cards_state = np.sum(self.community_cards, axis=0)
        return np.concatenate((community_cards_state, agent.hand.sum(axis=0), np.array([agent.chips])))
        
    def _get_info(self):
        return {}

    def set_up(self, seed=None, return_info=False, options=None):
        self.deck = np.arange(52)
        np.random.shuffle(self.deck)
        self.deck = self.deck.tolist()
        self.game_step = 0
        self.round_step = 0
        for a in self.all_agents:
            hand = np.zeros((2,52))
            hand[0,:] = Card(self.deck.pop()).vec
            hand[1,:] = Card(self.deck.pop()).vec
            a.hand = hand
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
        observation = self.get_obs(self.training_agent)
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        self.agent_step(self.training_agent, action)
        return self.simulate_until_next_turn()
        
    def agent_step(self, agent, action):
        # Fold
        if action == 0:
           return self.fold(agent)
        # Call
        elif action == 1:
            call_size = self.bet_size - agent.round_bet
            if call_size > agent.chips:
                return self.fold(agent)
            else:
                agent.chips -= call_size
                agent.round_bet += call_size
                agent.reward = 0
                self.pot_size += call_size
        # Raise
        else:
            bet_size = self.bet_size - agent.round_bet + 1
            if bet_size > agent.chips:
                return self.fold(agent)
            else:
                agent.chips-= bet_size
                agent.round_bet += bet_size
                agent.reward = 0
                self.pot_size += bet_size

    def fold(self, agent):
        agent.folded = 1
        agent.done = True
        agent.reward = -agent.round_bet 
        return self.get_obs(self.training_agent), agent.reward, True, {}

    def simulate_until_next_turn(self):
        for a in self.other_agents:
            if self.is_betting_round_over():
                self.progress_game_step()

            self.agent_step(a, np.random.randint(0,3))

        return self.get_obs(self.training_agent), self.training_agent.reward, self.training_agent.done, {}

    def is_betting_round_over(self):
        folded = np.array([self.all_agents[i].folded for i in range(self.num_agents)])
        mask = np.where(folded == 0)
        round_bets = np.array([self.all_agents[i].round_bet for i in range(self.num_agents)])
        bets = round_bets[mask]
        if bets.min() == bets.max(): 
            return True
        return False

    def progress_game_step(self):
        self.bet_size = 1
        self.game_step += 1
        if self.game_step == 1:
            for i in range(3):
                self.community_cards[i,:] = Card(self.deck.pop()).vec
        elif self.game_step == 2:
            self.community_cards[3,:] = Card(self.deck.pop()).vec
        elif self.game_step == 3:
            self.community_cards[4,:] = Card(self.deck.pop()).vec 
        else:
            self.showdown()

    def showdown(self):
        scores = np.zeros((6,3))
        for i, a in enumerate(self.all_agents):
            if not a.folded: 
                scores[i,:] = self.score_hand(a.hand)
        print(scores)


    def render(self, mode="ansi"):
        
        return ''

    def score_hand(self, hand):
        hand = np.concatenate((hand, self.community_cards), axis=0)
        hand = hand.reshape(7,4,13)
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
    
