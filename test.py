
from card import Card

import numpy as np

deck = np.arange(54)
np.random.shuffle(deck)
deck = deck.tolist()

board = np.zeros((5,52))
for i in range(5):
    board[i,:] = Card(deck.pop()).vec

hand = np.zeros((2,52))
for i in range(2):
    hand[i,:] = Card(deck.pop()).vec

all_cards = np.concatenate((board, hand))
all_cards = all_cards.reshape(7,4,13)

flush = all_cards.sum(axis=0).sum(axis=1)

is_flush = np.max(flush) >= 5
card_values = all_cards.sum(axis=0).sum(axis=0)
count = 0
trick_high = 0
hand_high = 0
is_stright = False
for i, s in enumerate(card_values): 
    if s >= 1:
        count += 1
        hand_high = i
        if count >=5:
            trick_high = i
            is_stright = True
    else:
        count = 0

print(all_cards.sum(axis=0))
        
if is_stright and is_flush:
    print('royal flush')
    print(trick_high)
is_four_of_a_kind = np.isin(card_values, 4)
if np.sum(is_four_of_a_kind):
    print('four of a kind')
    trick_high = np.argwhere(is_four_of_a_kind==True)[0][0]
    print(trick_high)
is_three_of_a_kind = np.isin(card_values, 3)
is_pair = np.isin(card_values, 2)
if np.sum(is_pair) >=1 and np.sum(is_three_of_a_kind):
    print('full house')
    trick_high = np.argwhere(is_three_of_a_kind==True)[0][0]
    print(trick_high)
if is_flush:
    print(all_cards.sum(axis=0)[np.argmax(flush)])
    trick_high = np.argwhere(all_cards.sum(axis=0)[np.argmax(flush)]==1)[-1][0]
    print('flush')
if is_stright:
    print('straight')
    print(trick_high)
if np.sum(is_three_of_a_kind):
    print('three of a kind')
    trick_high = np.argwhere(is_three_of_a_kind==True)[0][0]
    print(trick_high)
if  np.sum(is_pair) >= 2:
    print('two pair')
    trick_high = np.argwhere(is_pair==True)[1][0]
    print(trick_high)
if  np.sum(is_pair):
    print('pair')
    trick_high = np.argwhere(is_pair==True)[0][0]
    print(trick_high)

print('Trick high: ' + str(trick_high))
print('Hand high: ' + str(hand_high))
