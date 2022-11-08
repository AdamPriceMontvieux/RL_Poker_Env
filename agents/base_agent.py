import numpy as np

class BaseAgent:

    def __init__(self, ID):
        self.ID = int(ID)
        self.hand = np.ones(24)
        self.chips = 100
        self.folded = 0
        self.round_bet = 0
        self.game_bet = 0
        self.done = False
        self.reward_buffer = 0
    
    def get_hand(self):
        return self.hand.sum(axis=0)


