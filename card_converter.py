import numpy as np

class CardConverter():

    def __init__(self):
        self.suit_str_to_int = {'CLUBS': 0, 'DIAMONDS': 1, 'SPADES': 2, 'HEARTS': 3}
        self.value_str_to_int= {'ACE': 0, 'TWO': 1, 'THREE': 2,
                                'JACK': 3, 'QUEEN': 4, 'KING': 5}
        self.suit_int_to_str = {v: k for k, v in self.suit_str_to_int.items()}
        self.value_int_to_str = {v: k for k, v in self.value_str_to_int.items()}

    def vec_to_string(self, vector):
        if np.max(vector) == 0: return ''
        index = np.argmax(vector)
        self.suit = int(index / 6)
        self.value = index % 6
        return f"{self.value_int_to_str[self.value]} of {self.suit_int_to_str[self.suit]}"
