#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:43:43 2021

@author: labuser
"""


prior_probs = {
    'S1': 0.4,
    'S2': 0.3,
    'S3': 0.3,
}
transition_probs = {
    'S1': {'S1': 0.4, 'S2': 0.4, 'S3': 0.2},
    'S2': {'S1': 0.4, 'S2': 0.3, 'S3': 0.3},
    'S3': {'S1': 0.25, 'S2': 0.25, 'S3': 0.5}
}
states = ['S1', 'S2', 'S3']

#output 32,38,34
# Observation T1(𝑊 ∧ 𝑅 ∧ 𝑆𝐺); T2(¬𝑊 ∧ ¬𝑅 ∧ ¬𝑆𝐺); T2(𝑊 ∧ 𝑅 ∧ 𝑆𝐺) 
defult_prob = 1e-4
emission_probs = {'S1':[0.05,0.5,0.05],
                  'S2':[defult_prob,0.3,defult_prob],
                  'S3':[defult_prob,0.1,defult_prob]}   
alpha = {} #will be used to save the expected value (probs) of the previous evidence being in each state

evediences = 3






for i in range(evediences):
     
    if i == 0:
        alpha["evidence {}".format(i)] = {}
        for state in states:
            alpha["evidence {}".format(i)][state] = prior_probs[state] * emission_probs[state][0]
    else:
        
        alpha["evidence {}".format(i)] = {}

        for current_state in states: #only states for buy word
            sum_ = 0
            for previous_state in states:
                sum_ += alpha["evidence {}".format(i-1)][previous_state] *transition_probs[previous_state][current_state] 
            sum_ *= emission_probs[current_state][i]
            #save the highest probs and update its path
            alpha["evidence {}".format(i)][current_state] = sum_

prob_observation_given_model = sum(alpha["evidence {}".format(i)].values())


