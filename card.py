import numpy as np

class Card():

    def __init__(self, index):
        self.suit_str_to_int = {'CLUBS': 0, 'DIAMONDS': 1, 'SPADES': 2, 'HEARTS': 3}
        self.value_str_to_int= {'ACE': 0, 'TWO': 1, 'THREE': 2, 'FOUR': 3,
                                'FIVE': 4, 'JACK': 5, 'QUEEN': 6, 'KING': 7}
        self.suit_int_to_str = {v: k for k, v in self.suit_str_to_int.items()}
        self.value_int_to_str = {v: k for k, v in self.value_str_to_int.items()}

        self.suit = int(index / 6)
        self.value = index % 6

        self.vec = np.zeros(52)
        self.vec[index] = 1

    def asvector(self):
        vector = np.zeros(32)
        index = self.suit * 6 + self.value
        vector[index] = 1
        return vector

    def asstring(self):
        return f"{self.value_int_to_str[self.value]} of {self.suit_int_to_str[self.suit]}"

    def value_as_string(self):
        return self.value_int_to_str(self.value)

    def suit_as_string(self):
        return self.suit_int_to_str(self.suit)