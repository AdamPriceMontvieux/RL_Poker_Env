
from card import Card

import numpy as np

deck = np.arange(52)
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
    print('royal flush')
if np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 4)):
    print('four of a kind')
if np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 2)) and np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 3)):
    print('full house')
if is_flush:
    print('flush')
if is_stright:
    print('straight')
if np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 3)):
    print('three of a kind')
if np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 2)) >= 2:
    print('two pair')
if np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 2)):
    print('pair')
print(high)

print(all_cards.sum(axis=0))
four_of_a_kind = np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 4))
full_house = np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 2)) and np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 3))
three_of_a_kind = np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 3))
two_pair = np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 2)) >= 2
pair = np.sum(np.isin(all_cards.sum(axis=0).sum(axis=0), 2))
