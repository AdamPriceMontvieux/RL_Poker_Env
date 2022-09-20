import numpy as np

class BaseAgent:

    def __init__(self, ID):
        self.ID = int(ID)
        self.hand = np.ones(52)
        self.chips = 100
        self.folded = 0
        self.round_bet = 0
        self.done = False
        self.reward = 0
    
    def train(self):
        """allows an agent to learn a policy"""
        raise NotImplementedError

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's
         internal state and provided observation and action space"""
        raise NotImplementedError
    
    def process_observation(self, observation):
        """Process an observation vector"""
        raise NotImplementedError


