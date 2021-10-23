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

def make_job_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    BayesNet.add_node("GM")
    BayesNet.add_node("WR")
    BayesNet.add_node("WP")
    BayesNet.add_node("I")
    BayesNet.add_node("TA")
    BayesNet.add_node("JO")
    BayesNet.add_node("SC")
    
    BayesNet.add_edge("GM","I")
    BayesNet.add_edge("WR","I")
    BayesNet.add_edge("WP","I")
    BayesNet.add_edge("I","TA")
    BayesNet.add_edge("I","JO")
    BayesNet.add_edge("TA","JO")
    BayesNet.add_edge("TA","SC")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the job intreview net.
    """
    cpd_gm = TabularCPD("GM",2,values=[[0.3],[0.7]])
    cpd_wr = TabularCPD("WR",2,values=[[0.4],[0.6]])
    cpd_wp = TabularCPD("WP",2,values=[[0.2],[0.8]])

    cpd_i = TabularCPD("I",2,values = [[0.7,0.4,0.5,0.3,0.5,0.3,0.4,0.1],/
                                       [0.3,0.6,0.5,0.7,0.5,0.7,0.6,0.9]],
                       evidence=['GM', 'WR','WP'], evidence_card=[2,2,2])

    
    cpd_ta = TabularCPD('TA', 2, values=[[ 0.5,0.1], \
                    [ 0.5,0.9]], evidence=['I'], evidence_card=[2])
    

    cpd_sc = TabularCPD('SC', 2, values=[[ 0.8,0.3], \
                    [ 0.2,0.7]], evidence=['TA'], evidence_card=[2])

    cpd_jo = TabularCPD('JO', 2, values=[[0.7,0.3,0.4,0.1], \
                    [0.3,0.7,0.6,0.9]], evidence=['I', 'TA'], evidence_card=[2, 2])

                    
    bayes_net.add_cpds(cpd_gm, cpd_wr, cpd_wp,cpd_i,cpd_ta,cpd_sc,cpd_jo)
    return bayes_net


def get_gm(bayes_net):
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['GM'], joint=False)
    GM = marginal_prob['GM'].values
    return alarm_prob[1]


#part 1 
def get_I_given_WP_WR(bayes_net):
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['I'],evidence={'WP': 1,'WR':1}, joint=False)
    i_prob = conditional_prob['I'].values
    return i_prob[1]
#part 2 
def get_SC_given_I_JO(bayes_net):
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['SC'],evidence={'I': 1,'JO':1}, joint=False)
    sc_prob = conditional_prob['SC'].values
    return sc_prob[1]
#part 3
def get_TA_given_WP_WR(bayes_net):
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['TA'],evidence={'WP': 1,'WR':1}, joint=False)
    ta_prob = conditional_prob['TA'].values
    return ta_prob[0]

#part 4
def check_independence(bayes_net):
    #option 1
    answers ={}
    solver = VariableElimination(bayes_net)
    JO_given_TA_WR = solver.query(variables=['JO'],evidence={'TA': 1,'WR':1}, joint=False)
    jo_prob1 = JO_given_TA_WR['JO'].values
    
    JO_given_TA = solver.query(variables=['JO'],evidence={'TA': 1}, joint=False)
    jo_prob2 = JO_given_TA['JO'].values
    answers['1'] = (jo_prob1==jo_prob2).all()
    
    #option 3
    JO_given_I = solver.query(variables=['JO'],evidence={'I': 1}, joint=False)
    jo_prob1 = JO_given_I['JO'].values
    
    JO_given_notI = solver.query(variables=['JO'],evidence={'I': 0}, joint=False)
    jo_prob2 = JO_given_notI['JO'].values
    
    answers['3'] = (jo_prob1==jo_prob2).all()
    
    #optoin4 
    sc_given_I_WP = solver.query(variables=['SC'],evidence={'I': 1,'WP':1}, joint=False)
    sc_prob1 = sc_given_I_WP['SC'].values
    
    sc_given_I = solver.query(variables=['SC'],evidence={'I': 1}, joint=False)
    sc_prob2 = sc_given_I['SC'].values
    answers['4'] = (sc_prob1==sc_prob2).all()
    
    #option5
    TA_given_I_GM = solver.query(variables=['TA'],evidence={'I': 1,'GM':1}, joint=False)
    ta_prob1 = TA_given_I_GM['TA'].values
    
    TA_given_I = solver.query(variables=['TA'],evidence={'I': 1}, joint=False)
    ta_prob2 = TA_given_I['TA'].values
    answers['5'] = (ta_prob1==ta_prob2).all()
    

    
    return answers


