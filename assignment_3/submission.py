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

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")

    BayesNet.add_edge("temperature","gauge")
    BayesNet.add_edge("temperature","faulty gauge")
    BayesNet.add_edge("faulty gauge","gauge")
    BayesNet.add_edge("gauge","alarm")
    BayesNet.add_edge("faulty alarm","alarm")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    cpd_t = TabularCPD("temperature",2,values=[[0.8],[0.2]])

    cpd_fg_t = TabularCPD('faulty gauge', 2, values=[[ 0.95,0.2], \
                    [ 0.05,0.8]], evidence=['temperature'], evidence_card=[2])

    cpd_g_t_fg = TabularCPD('gauge', 2, values=[[0.95,0.2,0.05,0.8], \
                    [0.05,0.8,0.95,0.2]], evidence=['temperature', 'faulty gauge'], evidence_card=[2, 2])

    cpd_fa = TabularCPD("faulty alarm",2,values=[[0.85],[0.15]])

    cpd_a_g_fa = TabularCPD('alarm', 2, values=[[0.9, 0.55, 0.1, 0.45], \
                    [0.1, 0.45, 0.9, 0.55]], evidence=['gauge', 'faulty alarm'], evidence_card=[2, 2])
                    
    bayes_net.add_cpds(cpd_t, cpd_fg_t, cpd_g_t_fg,cpd_fa,cpd_a_g_fa)
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    alarm_prob = marginal_prob['alarm'].values
    return alarm_prob[1]


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    gauge_prob = marginal_prob['gauge'].values
    return gauge_prob[1]


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'],evidence={'alarm': 1,'faulty alarm':0,'faulty gauge':0}, joint=False)
    temp_prob = conditional_prob['temperature'].values
    return temp_prob[1]


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")

    BayesNet.add_edge("A","AvB")
    BayesNet.add_edge("B","AvB")
    BayesNet.add_edge("B","BvC")
    BayesNet.add_edge("C","BvC")
    BayesNet.add_edge("C","CvA") 
    BayesNet.add_edge("A","CvA") 
    
    #set propability 
    cpd_a = TabularCPD("A",4,values=[[0.15],[0.45],[0.3],[0.1]])
    cpd_b = TabularCPD("B",4,values=[[0.15],[0.45],[0.3],[0.1]])
    cpd_c = TabularCPD("C",4,values=[[0.15],[0.45],[0.3],[0.1]])
    
    cpd_a_v_b = TabularCPD('AvB', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], 
                                             [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                             [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]],
                                    evidence=['A', 'B'], evidence_card=[4, 4])
    
    cpd_c_v_b = TabularCPD('BvC', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], 
                                             [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                             [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]],
                                    evidence=['B', 'C'], evidence_card=[4, 4])
    
    cpd_c_v_a = TabularCPD('CvA', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], 
                                             [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                             [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]],
                                    evidence=['C', 'A'], evidence_card=[4, 4])
    
    BayesNet.add_cpds(cpd_a,cpd_b,cpd_c,cpd_a_v_b,cpd_c_v_b,cpd_c_v_a)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'],evidence={'AvB':0,'CvA':2}, joint=False)
    posterior = conditional_prob['BvC'].values
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    variables = ['A','B','C','AvB','BvC','CvA'] # the evidence are 'AvB' and 'CvA' -->fixed
    current_state_dict = {}
    #sample = [] #represent the value for each variable in the current sample
    
    team_skills = [0,1,2,3]
    match_result =[0,1,2]
    variable2choose = [0,1,2,4] # 'AvB' and 'CvA' are fixed
    
   
    for i in range(len(initial_state)): #initial_state must be of a length of 6    
        if i == 3:
            current_state_dict[variables[i]] = 0 # A beats B
        elif i==5:
            current_state_dict[variables[i]] = 2 # A draws with C (fixed)
        else:
            if initial_state[i] is None:
                if i == 4:    
                    current_state_dict[variables[i]] = np.random.choice(match_result) #BvC is choicen randomly 
                else:
                     current_state_dict[variables[i]] = np.random.choice(team_skills) #A,B, and C are choicen randomly                    
            else:
                current_state_dict[variables[i]]= initial_state[i]
  
        
        
   
    #choose a state to change at random
    chosen_variable = np.random.choice(variable2choose) 
    if variables[chosen_variable] == 'A':
        normalizer = 0
        prob=[]
            
        #now calculate unormalized probaility for each skill for a, then normailize
        for skill in team_skills:
            p_A_skill = bayes_net.get_cpds('A').values[skill]      
            p_AvB_bar_A_B = bayes_net.get_cpds("AvB").values[current_state_dict['AvB']][skill][current_state_dict['B']]
            p_CvA_bar_C_A = bayes_net.get_cpds("CvA").values[current_state_dict['CvA']][current_state_dict['C']][skill]
            p= p_A_skill*p_AvB_bar_A_B*p_CvA_bar_C_A 
            prob.append(p)
            normalizer += p
        prob = [p/normalizer for p in prob]
            
        current_state_dict['A']= np.random.choice(team_skills,1,p=prob)[0]
            
            
    elif  variables[chosen_variable] == 'B':
        normalizer = 0
        prob=[]
        
        #now calculate unormalized probaility for each skill for a, then normailize
        for skill in team_skills:
            p_B_skill = bayes_net.get_cpds('B').values[skill]      
            p_AvB_bar_A_B = bayes_net.get_cpds("AvB").values[current_state_dict['AvB']][current_state_dict['A']][skill]
            p_BvC_bar_B_C = bayes_net.get_cpds("BvC").values[current_state_dict['BvC']][skill][current_state_dict['C']]
            p = p_B_skill*p_AvB_bar_A_B*p_BvC_bar_B_C
            #normalize
            prob.append(p)
            normalizer += p
        prob = [p/normalizer for p in prob]
        
        current_state_dict['B']= np.random.choice(team_skills,1,p=prob)[0]
            
    elif  variables[chosen_variable] == 'C':
        normalizer = 0
        prob=[]
        for skill in team_skills:
            p_C_skill = bayes_net.get_cpds('C').values[skill]      
            p_BvC_bar_B_C = bayes_net.get_cpds("BvC").values[current_state_dict['BvC']][current_state_dict['B']][skill]
            p_CvA_bar_C_A = bayes_net.get_cpds("CvA").values[current_state_dict['CvA']][skill][current_state_dict['A']]
            p= p_C_skill*p_BvC_bar_B_C*p_CvA_bar_C_A
            prob.append(p)
            normalizer += p
        prob = [p/normalizer for p in prob]
        current_state_dict['C']= np.random.choice(team_skills,1,p=prob)[0]
    elif  variables[chosen_variable] == 'BvC':
        prob = []
        #this can be pulled out from the table directed as BvC only directly depends on B and C
        for result in match_result:
            p = bayes_net.get_cpds('BvC').values[result][current_state_dict['B']][current_state_dict['C']]
            #no need to normalize as the total_prob is 1
            prob.append(p)
            
        current_state_dict['BvC']= np.random.choice(match_result,1,p=prob)[0]
            
            
    #extract the sample in a list
    sample = [current_state_dict['A'],current_state_dict['B'],current_state_dict['C'],current_state_dict['AvB'],current_state_dict['BvC'],current_state_dict['CvA']]
        
    sample = tuple(sample)    
    return sample

def test_Gibbes(bayes_net,sample,n):
    B_beats_C = 0.0
    C_beats_B = 0.0
    draw =0.0
    for i in range(n):
        sample = Gibbs_sampler(bayes_net=bayes_net,initial_state=sample)
        if sample[4]==0:
            B_beats_C +=1.
        elif sample[4]==1:
            C_beats_B +=1.
        elif sample[4]==2:
            draw += 1.0
    
    return [B_beats_C/n,C_beats_B/n,draw/n]
            
    
    
        


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    variables = ['A','B','C','AvB','BvC','CvA'] # the evidence are 'AvB' and 'CvA' -->fixed
    #sample = [] #represent the value for each variable in the current sample
    current_sample_dict = {}
    old_sample_dict = {}
    team_skills = [0,1,2,3]
    match_result =[0,1,2]
    

    #check it is the first sample
    if initial_state[0] is None:
        for i in range(len(initial_state)): #initial_state must be of a length of 6    
            if i == 3:
                current_sample_dict[variables[i]] = 0 # A beats B
            elif i==5:
                current_sample_dict[variables[i]] = 2 # A draws with C (fixed)  
            elif i == 4:    
                current_sample_dict[variables[i]] = np.random.choice(match_result) #BvC is choicen randomly 
            else:
                 current_sample_dict[variables[i]] = np.random.choice(team_skills) #A,B, and C are choicen randomly  
        sample = [current_sample_dict['A'],current_sample_dict['B'],current_sample_dict['C'],current_sample_dict['AvB'],current_sample_dict['BvC'],current_sample_dict['CvA']]
        
        sample = tuple(sample)    
        return sample 
    
    #compute the propabiliy the of the old sample 
    for i in range(len(initial_state)):
        old_sample_dict[variables[i]]= initial_state[i]
    
    p_A = bayes_net.get_cpds('A').values[old_sample_dict['A']] 
    p_B = bayes_net.get_cpds('B').values[old_sample_dict['B']] 
    p_C = bayes_net.get_cpds('C').values[old_sample_dict['C']] 
    p_AvB_bar_A_B = bayes_net.get_cpds("AvB").values[old_sample_dict['AvB']][old_sample_dict['A']][old_sample_dict['B']]
    p_BvC_bar_A_C = bayes_net.get_cpds("BvC").values[old_sample_dict['BvC']][old_sample_dict['B']][old_sample_dict['C']]
    p_CvA_bar_C_A = bayes_net.get_cpds("CvA").values[old_sample_dict['CvA']][old_sample_dict['C']][old_sample_dict['A']] 
    joint_old = p_A*p_B*p_C*p_AvB_bar_A_B*p_BvC_bar_A_C*p_CvA_bar_C_A
    
    #create an independent second sample 
    for i in range(len(initial_state)): #initial_state must be of a length of 6    
            if i == 3:
                current_sample_dict[variables[i]] = 0 # A beats B
            elif i==5:
                current_sample_dict[variables[i]] = 2 # A draws with C (fixed)  
            elif i == 4:    
                current_sample_dict[variables[i]] = np.random.choice(match_result) #BvC is choicen randomly 
            else:
                 current_sample_dict[variables[i]] = np.random.choice(team_skills) #A,B, and C are choicen randomly  
    #calculate the current sample 
    p_A = bayes_net.get_cpds('A').values[current_sample_dict['A']] 
    p_B = bayes_net.get_cpds('B').values[current_sample_dict['B']] 
    p_C = bayes_net.get_cpds('C').values[current_sample_dict['C']] 
    p_AvB_bar_A_B = bayes_net.get_cpds("AvB").values[current_sample_dict['AvB']][current_sample_dict['A']][current_sample_dict['B']]
    p_BvC_bar_A_C = bayes_net.get_cpds("BvC").values[current_sample_dict['BvC']][current_sample_dict['B']][current_sample_dict['C']]
    p_CvA_bar_C_A = bayes_net.get_cpds("CvA").values[current_sample_dict['CvA']][current_sample_dict['C']][current_sample_dict['A']] 
    joint_current= p_A*p_B*p_C*p_AvB_bar_A_B*p_BvC_bar_A_C*p_CvA_bar_C_A
    
    alpha = min(1,joint_current/joint_old)
    
    #if the new sample has a joint probability more than the old, it will be chosen with probabiliyt of 1
    #otherwise, it will be only accepted with probability of alpha
    chosen_sample = np.random.choice(['new','old'],1,p=[alpha,1-alpha])
    
    if chosen_sample == 'new':
        sample = [current_sample_dict['A'],current_sample_dict['B'],current_sample_dict['C'],current_sample_dict['AvB'],current_sample_dict['BvC'],current_sample_dict['CvA']]
        sample = tuple(sample)
    else:
        sample = [old_sample_dict['A'],old_sample_dict['B'],old_sample_dict['C'],old_sample_dict['AvB'],old_sample_dict['BvC'],old_sample_dict['CvA']]
        sample = tuple(sample)
    
    return sample
        
def test_MH(bayes_net,sample,n):
    B_beats_C = 0.0
    C_beats_B = 0.0
    draw =0.0
    for i in range(n):
        sample = MH_sampler(bayes_net=bayes_net,initial_state=sample)
        if sample[4]==0:
            B_beats_C +=1.
        elif sample[4]==1:
            C_beats_B +=1.
        elif sample[4]==2:
            draw += 1.0
    
    return [B_beats_C/n,C_beats_B/n,draw/n]
                         
    


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    
    delta = 0.0001
    max_iteration = 30000
    N = 100 #check for delta every N
    #initalize random posterior
    Gibbs_convergence = [0.33,0.33,0.33] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0.33,0.33,0.33] # posterior distribution of the BvC match as produced by MH
    
    old_posterior_gibbs =[0,0,0]
    old_posterior_MH =[0,0,0]
    
    #test gibbs
    sample_gibbs = initial_state
    B_beats_C = 0.0
    C_beats_B = 0.0
    draw =0.0
    while Gibbs_count <max_iteration:
        Gibbs_count += 1
        sample_gibbs = MH_sampler(bayes_net=bayes_net,initial_state=sample_gibbs)
        if sample_gibbs[4]==0:
            B_beats_C +=1.
        elif sample_gibbs[4]==1:
            C_beats_B +=1.
        elif sample_gibbs[4]==2:
            draw += 1.0
        if (Gibbs_count%N)==0:
            old_posterior_gibbs = Gibbs_convergence
            Gibbs_convergence = [B_beats_C/Gibbs_count,C_beats_B/Gibbs_count,draw/Gibbs_count]
            diff_avg =  (abs(old_posterior_gibbs[0]-Gibbs_convergence[0]) + abs(old_posterior_gibbs[1]-Gibbs_convergence[1]) + abs(old_posterior_gibbs[2]-Gibbs_convergence[2]))/3
            if diff_avg < delta:
                break
    
    #test MH
    sample_MH = initial_state
    B_beats_C = 0.0
    C_beats_B = 0.0
    draw =0.0
    
    while MH_count < max_iteration:
        MH_count += 1
        old_sample_HM = sample_MH
        sample_MH = MH_sampler(bayes_net=bayes_net,initial_state=sample_MH)
        if old_sample_HM == sample_MH:
            MH_rejection_count+=1
        if sample_MH[4]==0:
            B_beats_C +=1.
        elif sample_MH[4]==1:
            C_beats_B +=1.
        elif sample_MH[4]==2:
            draw += 1.0
        if (MH_count%N)==0:
            old_posterior_MH = MH_convergence
            MH_convergence = [B_beats_C/MH_count,C_beats_B/MH_count,draw/MH_count]
            diff_avg = (abs(old_posterior_MH[0]-MH_convergence[0])+ abs(old_posterior_MH[1]-MH_convergence[1]) + abs(old_posterior_MH[2]-MH_convergence[2]))/3
            if diff_avg < delta:
                break
    #factor = Gibbs_count/MH_count
    #print("factor: ", Gibbs_count/MH_count)   
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

# def average_factor(bayes_net,initial_state,N):
#     factor_sum = 0.0
#     for i in range(N):
#         sample = initial_state
#         Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count,factor = compare_sampling(bayes_net,sample)
#         factor_sum += factor
    
#     return factor_sum/N

def sampling_question():
    """Question about sampling performance."""
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.1
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Ali Alrasheed"
