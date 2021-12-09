#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:29:54 2021

@author: labuser
"""

import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_net():
    
    BayesNet = BayesianModel()
    BayesNet.add_node("Smoke")
    BayesNet.add_node("Unhealthy")
    BayesNet.add_node("Lung_cancer")
    BayesNet.add_node("Breathing")
    BayesNet.add_node("Rash")

    BayesNet.add_edge("Smoke","Lung_cancer")
    BayesNet.add_edge("Unhealthy","Lung_cancer")
    BayesNet.add_edge("Lung_cancer","Breathing")
    BayesNet.add_edge("Lung_cancer","Rash")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    cpd_s= TabularCPD("Smoke",2,values=[[0.7],[0.3]])
    cpd_u= TabularCPD("Unhealthy",2,values=[[0.6],[0.4]])
    cpd_l_given_su = TabularCPD('Lung_cancer', 2, values=[[0.95,0.92,0.9,0.85], \
                    [0.05,0.08,0.1,0.15]], evidence=['Smoke', 'Unhealthy'], evidence_card=[2, 2])

    cpd_r_given_l = TabularCPD('Rash', 2, values=[[ 0.97,0.9], \
                    [ 0.03,0.1]], evidence=['Lung_cancer'], evidence_card=[2])
        
    cpd_b_given_l = TabularCPD('Breathing', 2, values=[[ 0.95,0.4], \
                    [ 0.05,0.6]], evidence=['Lung_cancer'], evidence_card=[2])

    

    
                    
    bayes_net.add_cpds(cpd_s, cpd_u, cpd_l_given_su,cpd_r_given_l,cpd_b_given_l)
    return bayes_net


def get_prob(bayes_net):
    """Calculate the marginal 
    probability of person over 60 that he/she does not have cancer given they have severe breathing but not Rash"""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['Lung_cancer'],evidence={'Breathing':1,'Rash':0}, joint=False)
    cancer_prob = marginal_prob['Lung_cancer'].values
    return cancer_prob

bayes_net = make_net()
bayes_net = set_probability(bayes_net)
prob = get_prob(bayes_net) #[False, True]




