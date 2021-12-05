# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 15:38:05 2021

@author: Labuser
"""

import numpy as np
"""Word BUY"""
b_prior_probs = {
    'B1': 0.333,
    'B2': 0.,
    'B3': 0.,
    'Bend': 0.,
}
b_transition_probs = {
    'B1': {'B1': 0.625, 'B2': 0.375, 'B3': 0., 'Bend': 0.},
    'B2': {'B1': 0., 'B2': 0.625, 'B3': 0.375, 'Bend': 0.},
    'B3': {'B1': 0., 'B2': 0., 'B3': 0.625, 'Bend': 0.375},
    'Bend': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 1.},
}
# Parameters for end state is not required
b_emission_paras = {
    'B1': (41.750, 2.773),
    'B2': (58.625, 5.678),
    'B3': (53.125, 5.418),
    'Bend': (None, None)
}

"""Word CAR"""
c_prior_probs = {
    'C1': 0.333,
    'C2': 0.,
    'C3': 0.,
    'Cend': 0.,
}
c_transition_probs = {
    'C1': {'C1': 0.667, 'C2': 0.333, 'C3': 0., 'Cend': 0.},
    'C2': {'C1': 0., 'C2': 0., 'C3': 1., 'Cend': 0.},
    'C3': {'C1': 0., 'C2': 0., 'C3': 0.8, 'Cend': 0.2},
    'Cend': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 1.},
}
# Parameters for end state is not required
c_emission_paras = {
    'C1': (35.667, 4.899),
    'C2': (43.667, 1.700),
    'C3': (44.200, 7.341),
    'Cend': (None, None)
}

"""Word HOUSE"""
h_prior_probs = {
    'H1': 0.333,
    'H2': 0.,
    'H3': 0.,
    'Hend': 0.,
}
# Probability of a state changing to another state.
h_transition_probs = {
    'H1': {'H1': 0.667, 'H2': 0.333, 'H3': 0., 'Hend': 0.},
    'H2': {'H1': 0., 'H2': .857, 'H3': 0.143, 'Hend': 0.},
    'H3': {'H1': 0., 'H2': 0., 'H3': 0.813, 'Hend': 0.187},
    'Hend': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 1.},
}
# Parameters for end state is not required
h_emission_paras = {
    'H1': (45.333, 3.972),
    'H2': (34.952, 8.127),
    'H3': (67.438, 5.733),
    'Hend': (None, None)
}

evidence_vector = [44, 51, 57, 63, 61, 60, 59]
b_states = ['B1', 'B2', 'B3', 'Bend']
c_states = ['C1', 'C2', 'C3', 'Cend']
h_states = ['H1', 'H2', 'H3', 'Hend']


states = b_states + c_states + h_states

prior_probs = b_prior_probs
prior_probs.update(c_prior_probs)
prior_probs.update(h_prior_probs)

transition_probs = b_transition_probs
transition_probs.update(c_transition_probs)
transition_probs.update(h_transition_probs)

emission_paras = b_emission_paras
emission_paras.update(c_emission_paras)
emission_paras.update(h_emission_paras)
def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    Note:
        We simplify the problem so you don't have to take care of integrals.
        Theoretically speaking, the returned value is not a probability of x,
        since the probability of any single value x from a continuous 
        distribution should be zero, instead of the number outputed here.
        By definition, the Gaussian percentile of a given value "x"
        is computed based on the "area" under the curve, from left-most to x. 
        The proability of getting value "x" is zero bcause a single value "x"
        has zero width, however, the probability of a range of value can be
        computed, for say, from "x - 0.1" to "x + 0.1".

    """
    if list(para_tuple) == [None, None]:
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile


def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.
    Args:
        evidence_vector (list): List of right hand Y-axis positions (interger).
        states (list): List of all states in a word. No transition between words.
                       example: ['B1', 'B2', 'B3', 'Bend', 'H1', 'H2', 'H3', 'Hend']
        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}
        transition_probs (dict): dictionary representing transitions from each
                                 state to every other valid state such as for the above 
                                 states, there won't be a transition from 'B1' to 'H1'
        emission_paras (dict): parameters of Gaussian distribution 
                                from each state.
    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )
    Note:
        You are required to use the function gaussian_prob to compute the
        emission probabilities.
    """
    
    #check if the evidence is empty
    if len(evidence_vector) == 0:
        sequence = []
        probability = 0.0
        return (sequence,probability)
    
    vitebi_table = {} #will be used to save the expected value (probs) of the previous evidence being in each state
    backtrack = {} 
    emission_probs = {}
    
    #Get the emission probabilities for all states based on provided Gaussian
    for state in states:
        emission_probs[state] = []
        for evidence in evidence_vector:
            emission_probs[state].append(gaussian_prob(evidence,emission_paras[state]))
    
    #for each obeservation, loop through each "word", fill up the Table with probabilities while
    #tracking which state resulted in highest probability
    for i in range(len(evidence_vector)):
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
            for current_state in states[:4]: #only states for buy word
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
                
            
            #for Car word
            for current_state in states[4:8]: #only states for buy word
                p_max = 0
                state_max = None #the expected state (that has the highest expected value)
                for previous_state in states[4:8]:
                    p = vitebi_table["evidence {}".format(i-1)][previous_state] *transition_probs[previous_state][current_state] *emission_probs[current_state][i]
                    if p >= p_max:
                        p_max = p
                        state_max = previous_state #keeping track of the biterbi path
                #save the highest probs and update its path
                vitebi_table["evidence {}".format(i)][current_state] = p_max
                tem_backtrack[current_state] = backtrack[state_max] + [current_state]
                
            
            #for House word
            for current_state in states[8:]: #only states for buy word
                p_max = 0
                state_max = None #the expected state (that has the highest expected value)
                for previous_state in states[8:]:
                    p = vitebi_table["evidence {}".format(i-1)][previous_state] *transition_probs[previous_state][current_state] *emission_probs[current_state][i]
                    if p >= p_max:
                        p_max = p
                        state_max = previous_state #keeping track of the biterbi path
                #save the highest probs and update its path
                vitebi_table["evidence {}".format(i)][current_state] = p_max      
                tem_backtrack[current_state] = backtrack[state_max] + [current_state]
                
            backtrack = tem_backtrack #reassign it to backtrack

    #find the max probability and state in the end of the table
    max_state, max_probability = max(vitebi_table["evidence {}".format(len(evidence_vector)-1)].items(), key=lambda k: k[1])
    sequence = backtrack[max_state]
    
    if max_probability == 0:
        return  [], 0

    return sequence, max_probability
        
                
                        
                
            

