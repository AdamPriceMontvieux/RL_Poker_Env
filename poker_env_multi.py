
import gym
from gym import spaces
import numpy as np
#from sqlalchemy import false
from card import Card
from card_converter import CardConverter
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from typing import Dict, Optional
from collections import Iterable
from agents.base_agent import BaseAgent
from ray.rllib.utils import override

class PokerEnvMulti(MultiAgentEnv, gym.Env):

    NUM_AGENTS = 6
    NUM_ACTIONS = 3
    ACTION_SPACE = spaces.Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = None

    def __init__(self,  config: Optional[Dict] = None):

        if config is None:
            config = {}
        self._validate_config(config)
        self._load_config(config)

        self.deck = np.arange(28)
        self.community_cards = np.zeros((5, 28))
        self.hand_values = {0: 'High Card', 1: 'Pair', 2: 'Two Pair', 3: 'Three Of A Kind',
        4: 'Straight', 5: 'Flush', 6: 'Full House', 7: 'Four Of A Kind', 8: 'Stright Flush'}
        self.game_step = 0
        self.round_step = 0
        self.dealer = 0
        self.small_blind = 1
        self.big_blind = 2
        self.current_actor = 3
        self.game_number = -1
        self.pot_size = 0
        self.bet_size = 0
        self.raise_count = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(0, 500, shape=(29, )),
            ENV_STATE: spaces.Box(0, 1, shape=(28, ))
        })
        self.winner = []
        self.card_converter = CardConverter()
        self.info = False
        self.done = False

        self.agents = {}
        for a in self.players_ids:
            self.agents[a] = BaseAgent(a)

    def _validate_config(self, config):
        if "players_ids" in config:
            assert isinstance(config["players_ids"], Iterable)
            assert len(config["players_ids"]) == self.NUM_AGENTS

    def _load_config(self, config):
        self.players_ids = config.get("players_ids", [0,1,2,3,4,5])
        

    def get_obs(self, agent):
        community_cards_state = np.sum(self.community_cards, axis=0)
        return {"obs": self.agents[agent].get_obs(), ENV_STATE: community_cards_state} 

    def get_reward(self, agent):
        r = self.agents[agent].reward_buffer
        self.agents[agent].reward_buffer = 0
        return r
        
    def _get_info(self):
        return {}

    def set_up(self):
        self.deck = np.arange(28)
        np.random.shuffle(self.deck)
        self.deck = self.deck.tolist()
        self.game_step = 0
        self.round_step = 0
        for a in self.players_ids:
            hand = np.zeros((2,28))
            hand[0,:] = Card(self.deck.pop()).vec
            hand[1,:] = Card(self.deck.pop()).vec
            self.agents[a].hand = hand
            self.agents[a].game_bet = 0
            self.agents[a].round_bet = 0
            self.agents[a].done = False
            self.agents[a].folded = False

        self.game_number += 1
        self.dealer = self.game_number % self.NUM_AGENTS
        small_blind = (self.game_number+1) % self.NUM_AGENTS
        self.agents[small_blind].chips -= 1
        self.agents[small_blind].round_bet += 1
        big_blind = (self.game_number+2) % self.NUM_AGENTS
        self.agents[big_blind].chips -= 2
        self.agents[big_blind].round_bet += 2
        self.current_actor = (self.game_number+3) % self.NUM_AGENTS
        self.pot_size = 3
        self.bet_size = 1
        self.raise_count = 0
        self.winner = []
        self.done = False

    def reset(self, seed=None, return_info=False, options=None):
        self.set_up()
        obs = {self.current_actor: self.get_obs(self.current_actor)}
        info = self._get_info()
        return (obs, info) if return_info else obs

    @override(gym.Env)
    def step(self, action_dict):
        print(action_dict)
        self.agent_step(self.agents[self.current_actor], action_dict[self.current_actor])
        not_in_player = True
        while not_in_player:
            self.round_step += 1
            self.current_actor = (self.game_number+3+self.round_step) % self.NUM_AGENTS
            not_in_player = self.agents[self.current_actor].done
            if self.done: break

        if self.is_betting_round_over():
            self.progress_game_step()

        folded = np.array([self.agents[i].folded for i in range(self.NUM_AGENTS)])
        mask = np.array(np.where(folded == 0))
        print(mask[0].shape[0] )
        if mask[0].shape[0] == 1:
            self.done = True
            self.agents[self.current_actor].reward_buffer += self.pot_size
            self.agents[self.current_actor].chips += self.pot_size

        if self.done:
            print('done')
            print(self.get_info())
            empty_obs = {'obs': np.zeros(29), 'state': np.zeros(28)} 
            obs = {}; rewards = {}; dones = {}
            for a in self.players_ids:
                obs[a] = empty_obs
                rewards[a] = self.get_reward(a)
                dones[a] = True
            dones['__all__'] = True
        else:
            obs = {self.current_actor: self.get_obs(self.current_actor)}
            rewards = {self.current_actor: self.get_reward(self.current_actor)}
            dones = {self.current_actor: self.agents[self.current_actor].done, '__all__': False}
        
        return obs, rewards, dones, {}

    def _next_agent(self):
        return self.agents[self.current_actor]

    def agent_step(self, agent, action):
        # Fold
        if action == 0:
           self.fold(agent)
        # Call
        elif action == 1 or (action == 2 and self.raise_count >= 3):
            call_size = self.bet_size - agent.round_bet
            if call_size > agent.chips:
                self.fold(agent)
            else:
                agent.chips -= call_size
                agent.round_bet += call_size
                agent.reward = 0
                self.pot_size += call_size
        # Raise
        else:
            bet_size = self.bet_size - agent.round_bet + 1
            if bet_size > agent.chips:
                self.fold(agent)
            else:
                agent.chips -= bet_size
                agent.round_bet += bet_size
                agent.reward = 0
                self.pot_size += bet_size
                self.raise_count += 1

    def fold(self, agent):
        agent.folded = 1
        agent.done = True
        agent.reward_buffer = -agent.game_bet 

    def get_info(self):
        info = {}
        info['hands'] = [self.cards_to_string(self.agents[a].hand) for a in self.players_ids]
        info['community'] = self.cards_to_string(self.community_cards)
        scores = np.zeros((6,3))
        for i, a in enumerate(self.players_ids):
            if not self.agents[a].folded: 
                scores[i,:] = self.score_hand(self.agents[a].hand)
        info['scores'] = scores
        info['winners'] = self.winner
        return info

    def cards_to_string(self, card_vector):
        result = ''
        for i in range(card_vector.shape[0]):
            result += self.card_converter.vec_to_string(card_vector[i,:]) + '; '
        return result 

    def is_betting_round_over(self):
        folded = np.array([self.agents[i].folded for i in range(self.NUM_AGENTS)])
        mask = np.where(folded == 0)
        round_bets = np.array([self.agents[i].round_bet for i in range(self.NUM_AGENTS)])
        bets = round_bets[mask]
        if bets.min() == bets.max(): 
            return True
        return False

    def progress_game_step(self):
        print('progress')
        self.bet_size = 1
        self.game_step += 1
        self.raise_count = 0
        for a in self.players_ids:
            self.agents[a].game_bet += self.agents[a].round_bet 
            self.agents[a].round_bet = 0

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
        print('showdown')
        scores = np.zeros((6,3))
        for i, a in enumerate(self.players_ids):
            self.agents[a].done = True
            if not self.agents[a].folded: 
                scores[i,:] = self.score_hand(self.agents[a].hand)
        winners = np.arange(6)
        index = 0
        while winners.shape != () and index < 3:
            winners = winners[np.argwhere(scores[:,index][winners]==np.max(scores[:,index][winners])).squeeze()]
            scores[:, index]
            index += 1
        if winners.shape != ():
            for w in winners:
                self.agents[w].reward_buffer = int(self.pot_size / winners.shape[0])
                self.agents[w].chips += int(self.pot_size / winners.shape[0])
        else:
            self.agents[winners].reward_buffer = self.pot_size
            self.agents[winners].chips += self.pot_size
        losers = np.delete(np.arange(6), winners)
        for l in losers:
            self.agents[l].reward_buffer -= self.agents[l].game_bet
        self.winner = winners
        self.done = True


    def render(self, mode="ansi"):
        
        return ''

    def score_hand(self, hand):
        hand = np.concatenate((hand, self.community_cards), axis=0)
        hand = hand.reshape(-1,4,7)
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
    
