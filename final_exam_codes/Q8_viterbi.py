#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:11:30 2021

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
    
vitebi_table = {} #will be used to save the expected value (probs) of the previous evidence being in each state
backtrack = {} 
defult_prob = 1e-4
emission_probs = {'S1':[0.05,0.5,0.05],
                  'S2':[defult_prob,0.3,defult_prob],
                  'S3':[defult_prob,0.1,defult_prob]}
evediences = 3

#Get the emission probabilities for all states based on provided Gaussian
# for state in states:
#     emission_probs[state] = []
#     for evidence in evidence_vector:
#         emission_probs[state].append(gaussian_prob(evidence,emission_paras[state]))



#for each obeservation, loop through each "word", fill up the Table with probabilities while
#tracking which state resulted in highest probability
#for i in range(len(evidence_vector)):
for i in range(evediences):
     #use the prior probabilities to initialize Table for first observation (zero index of emission prob)
    if i == 0:
        vitebi_table["evidence {}".format(i)] = {}
        for state in states:
            vitebi_table["evidence {}".format(i)][state] = prior_probs[state] * emission_probs[state][0]
            backtrack[state] = [state]
    else:
        
        vitebi_table["evidence {}".format(i)] = {}

        tem_backtrack = {} #used to update the backtracking without overwriting it
        #for the word Buy
        for current_state in states: #only states for buy word
            p_max = 0
            state_max = None #the expected state (that has the highest expected value)
            for previous_state in states[:4]:
                p = vitebi_table["evidence {}".format(i-1)][previous_state] *transition_probs[previous_state][current_state] *emission_probs[current_state][i]
                if p >= p_max:
                    p_max = p
                    state_max = previous_state #keeping track of the biterbi path
            #save the highest probs and update its path
            vitebi_table["evidence {}".format(i)][current_state] = p_max
            tem_backtrack[current_state] = backtrack[state_max] + [current_state]
            
            
        backtrack = tem_backtrack #reassign it to backtrack

#find the max probability and state in the end of the table
max_state, max_probability = max(vitebi_table["evidence {}".format(evediences-1)].items(), key=lambda k: k[1])
sequence = backtrack[max_state]
    