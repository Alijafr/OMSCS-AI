#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:49:47 2021

@author: labuser
"""

import itertools, random
from scipy.stats import hypergeom

#calculate the expected value 
#note, the zero will not affect the calulation
#given that we draw one from pile_13, the only possible option is 3,2,1,0 
expected_value = 3*hypergeom.pmf(k=3,M=51, n=3, N=39)+2*hypergeom.pmf(k=2,M=51, n=3, N=39) + 1*hypergeom.pmf(k=1,M=51, n=3, N=39)+0*hypergeom.pmf(k=0,M=51, n=3, N=39)
expected_value_40 = expected_value +1 # the added ace from pile_13

#the probability now is just the expected value/num_cards
answer = expected_value_40/40

#verfiy answer below using sampling 


# make a deck of cards
deck = list(itertools.product(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'],
                              ['Spade','Heart','Diamond','Club'])) # will repeat 4 times for spade, heart, diamond ,and club

#probability to draw an ace from the 13 pile
N = 1000000
ace_drawn = 0
ace_drawn2 = 0
count = 0
expected_aces = 0
for trial in range(N):
    # shuffle the cards
    random.shuffle(deck)
    pile_13 = deck[:13]
    pile_39 = deck[13:]
    #aces = [d[0] for d in pile_39].count('A')
    #expected_aces += aces
    #card_drawn = random.shuffle(pile_13)
    if pile_13[0][0] == 'A':
        ace_drawn+=1
        #aces = [d[0] for d in pile_39].count('A')
        #expected_aces += aces
        pile_40 = pile_39 + [pile_13[0]]
        random.shuffle(pile_40)
        if pile_40[0][0] == 'A':
            ace_drawn2 +=1
    if pile_39[0][0] == 'A':
        count += 1
        
prob_13 = ace_drawn/N
prob_aces_drawn_twice = ace_drawn2 / N # result is 0.0045185 #whole event
expected_aces_39 = expected_aces/N

answer_sampled= ace_drawn2/ace_drawn #this is the answer




