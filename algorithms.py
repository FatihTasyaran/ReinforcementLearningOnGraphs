import graph_reader
import random
import os
import time
import matplotlib.pyplot as plt
import numpy as np

max_episode = 200
learning_rate = 0.9
discount = 0.9
epsilon = 1 ##Start with exploration
decay_rate = 1/(max_episode-(max_episode/4)) ##First half of episodes favor exploration,
                               ##Second half favor exploitation
max_q_depth = 1

def calculate_new_q(rewards, q_table, action, old_state, new_state):
    ##IMPLEMENTING BELLMAN OPTIMALITY
    old = (1-learning_rate)*q_table[old_state][action]

    max_q = -1000 ##Cardinal value to be changed

    for i in range(len(q_table[new_state])):
        if(q_table[new_state][i] != 'NA'):
            if(q_table[new_state][i] > max_q):
                max_q = q_table[new_state][i]

    new = learning_rate * (rewards[new_state] + discount*max_q)

    q_table[old_state][action] = old + new

def find_max_q(q_table, new_state):
    max_q = -1000 ##Cardinal value to be changed
    acting = -1000

    for i in range(len(q_table[new_state])):
        if(q_table[new_state][i] != 'NA'):
            if(q_table[new_state][i] > max_q):
                max_q = q_table[new_state][i]
                acting = i

    return max_q, acting

def calculate_new_q_with_depth(rewards, q_table, action, old_state, new_state, graph):
    ##IMPLEMENTING BELLMAN OPTIMALITY
    old = (1-learning_rate)*q_table[old_state][action]

    max_q = -1000 ##Cardinal value to be changed
    acting = -1000

    '''
    for i in range(len(q_table[new_state])):
        if(q_table[new_state][i] != 'NA'):
            if(q_table[new_state][i] > max_q):
                max_q = q_table[new_state][i]
                acting = i
    '''
    mimic = new_state
    mimic_q = 0
    for i in range(max_q_depth):
        max_q, acting = find_max_q(q_table, mimic)
        mimic_q += max_q
        act = dict(graph[mimic][0])
        act = {v: k for k, v in act.items()}

        #print('act:', act)                                                       
        #print('graph:', graph[mimic][0])
        #print('highest:', act[acting]) 
        
        mimic = act[acting]
        
        
    

    new = learning_rate * (rewards[new_state] + discount*(mimic_q/max_q_depth))
    q_table[old_state][action] = old + new
    

            
def choose_action(graph, q_table, state):
    global epsilon
    global decay_rate
    action = -1 ##Cardinal value to be changed
    
    rand_num = random.uniform(0,1)
    #print(rand_num)
    

    if rand_num > epsilon:
        #print('exploitation')
        my_max = -1000 ##Cardinal value to be changed
        for i in range(len(q_table[state])):
            if(q_table[state][i] != 'NA'):
                if(q_table[state][i] > my_max):
                    my_max = q_table[state][i]
                    index = i
                    #print('state:', state, 'index:', i)
        action = index ## q_table[i] stands for action i
            
    else:
        #print('exploration')
        possibles = []
        for i in range(len(q_table[state])):
            if(q_table[state][i] != 'NA'):
                possibles.append(i)

        action = random.choice(possibles)

    

    return action

def sarsa_calculate(graph, rewards, q_table, action, old_state, new_state):
    old = (1-learning_rate)*q_table[old_state][action]

    max_q = -1000 ##Cardinal value to be changed

    new_action = choose_action(graph, q_table, new_state)
    act = dict(graph[new_state][0])
    act = {v: k for k, v in act.items()}
    last_state = act[new_action]

    for i in range(len(q_table[last_state])):
        if(q_table[last_state][i] != 'NA'):
            if(q_table[last_state][i] > max_q):
                max_q = q_table[last_state][i]

    new = learning_rate * (rewards[new_state] + discount*max_q)

    q_table[old_state][action] = old + new

def q_learn(rewards, framework_graph, graph, q_table, start):
    #print('decay rate:', decay_rate)
    #print('epsilon goes to 0 in:', 1/decay_rate, ' episodes')

    global epsilon
    global decay_rate

    total_rewards = []
    found_in = []
    eps = []
    
    for episode in range(max_episode):
        print('****EPISODE ', episode, ' ****')
        state = start
        found = False
        time_step = 1
        while(not found):
            action = choose_action(graph, q_table, state)
            
            for le in range(len(graph[state][0])):
                #print('LE:', le)
                #print('length:', len(graph[state][0]))
                if(graph[state][0][le][1] == action):
                    new = graph[state][0][le][0]

            #print(state, '->', new)
            calculate_new_q(rewards, q_table, action, state, new)
            state = new
            
            if(framework_graph[state]['Faulty'] == True):
                found = True
                #print('FOUND!')
                print('Found in:', time_step, ' time steps')

            time_step += 1
                
        epsilon -= decay_rate
        #print(q_table)

        ####CALCULATING TOTAL REWARD AT THE END OF THE EPISODE####
        tot_reward = 0
        for i in range(len(q_table)):
            for j in range(len(q_table[0])):
                if(q_table[i][j] != 'NA'):
                    tot_reward += q_table[i][j]
        total_rewards.append(tot_reward)
        found_in.append(time_step)
        eps.append(epsilon)
        print('Total Reward:', tot_reward)
        ####CALCULATING TOTAL REWARD AT THE END OF THE EPISODE####


    ####DRAWING # OF TIME STEPS VS EPISODES AND REWARD####
    '''
    x = []
    for i in range(max_episode):
        x.append(i)
        
    plt.subplot(3,1,1)
    plt.plot(x, total_rewards, label = 'Total Reward', color='black')
    #plt.plot(x, found_in, label = 'Time Steps', color='red')
    plt.ylabel('Total Reward')

    plt.subplot(3,1,2)
    plt.plot(x, found_in, label = 'Time Steps', color='red')
    plt.ylabel('Found in Time Steps')

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='cyan')
    plt.ylabel('Epsilon Change')

    plt.tight_layout()
    plt.show()
    '''
    return total_rewards, found_in, eps
    ####DRAWING # OF TIME STEPS VS EPISODES AND REWARD####

def sarsa(rewards, framework_graph, graph, q_table, start):
    #print('decay rate:', decay_rate)
    #print('epsilon goes to 0 in:', 1/decay_rate, ' episodes')

    global epsilon
    global decay_rate

    total_rewards = []
    found_in = []
    eps = []
    
    for episode in range(max_episode):
        print('****EPISODE ', episode, ' ****')
        state = start
        found = False
        time_step = 1
        while(not found):
            action = choose_action(graph, q_table, state)
            
            for le in range(len(graph[state][0])):
                #print('LE:', le)
                #print('length:', len(graph[state][0]))
                if(graph[state][0][le][1] == action):
                    new = graph[state][0][le][0]

            #print(state, '->', new)
            sarsa_calculate(graph, rewards, q_table, action, state, new)
            state = new
            
            if(framework_graph[state]['Faulty'] == True):
                found = True
                #print('FOUND!')
                print('Found in:', time_step, ' time steps')

            time_step += 1
                
        epsilon -= decay_rate
        #print(q_table)

        ####CALCULATING TOTAL REWARD AT THE END OF THE EPISODE####
        tot_reward = 0
        for i in range(len(q_table)):
            for j in range(len(q_table[0])):
                if(q_table[i][j] != 'NA'):
                    tot_reward += q_table[i][j]
        total_rewards.append(tot_reward)
        found_in.append(time_step)
        eps.append(epsilon)
        print('Total Reward:', tot_reward)
        ####CALCULATING TOTAL REWARD AT THE END OF THE EPISODE####

    '''
    ####DRAWING # OF TIME STEPS VS EPISODES AND REWARD####
    x = []
    for i in range(max_episode):
        x.append(i)
        
    plt.subplot(3,1,1)
    plt.plot(x, total_rewards, label = 'Total Reward', color='black')
    #plt.plot(x, found_in, label = 'Time Steps', color='red')
    plt.ylabel('Total Reward')

    plt.subplot(3,1,2)
    plt.plot(x, found_in, label = 'Time Steps', color='red')
    plt.ylabel('Found in Time Steps')

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='cyan')
    plt.ylabel('Epsilon Change')

    plt.tight_layout()
    plt.show()
    '''

    return total_rewards, found_in, eps
    ####DRAWING # OF TIME STEPS VS EPISODES AND REWARD####


def q_learn_varying_depth(rewards, framework_graph, graph, q_table, start):
    #print('decay rate:', decay_rate)
    #print('epsilon goes to 0 in:', 1/decay_rate, ' episodes')

    global epsilon
    global decay_rate
    global max_q_depth

    total_rewards = []
    found_in = []
    eps = []
    
    for episode in range(max_episode):
        max_q_depth = int((max_episode/(episode+1))/4)
        if(max_q_depth == 0):
            max_q_depth = 1
        print('max_q_depth:', max_q_depth)
            
        print('****EPISODE ', episode, ' ****')
        state = start
        found = False
        time_step = 1
        while(not found):# and time_step < 100):
            action = choose_action(graph, q_table, state)
            
            for le in range(len(graph[state][0])):
                #print('LE:', le)
                #print('length:', len(graph[state][0]))
                if(graph[state][0][le][1] == action):
                    new = graph[state][0][le][0]

            #print(state, '->', new)
            calculate_new_q_with_depth(rewards, q_table, action, state, new, graph)
            state = new
            
            if(framework_graph[state]['Faulty'] == True):
                found = True
                #print('FOUND!')
                print('Found in:', time_step, ' time steps')

            time_step += 1
                
        epsilon -= decay_rate
        #print(q_table)

        ####CALCULATING TOTAL REWARD AT THE END OF THE EPISODE####
        tot_reward = 0
        for i in range(len(q_table)):
            for j in range(len(q_table[0])):
                if(q_table[i][j] != 'NA'):
                    tot_reward += q_table[i][j]
        total_rewards.append(tot_reward)
        found_in.append(time_step)
        eps.append(epsilon)
        print('Total Reward:', tot_reward)
        ####CALCULATING TOTAL REWARD AT THE END OF THE EPISODE####


    ####DRAWING # OF TIME STEPS VS EPISODES AND REWARD####
    '''
    x = []
    for i in range(max_episode):
        x.append(i)
        
    plt.subplot(3,1,1)
    plt.plot(x, total_rewards, label = 'Total Reward', color='black')
    #plt.plot(x, found_in, label = 'Time Steps', color='red')
    plt.ylabel('Total Reward')

    plt.subplot(3,1,2)
    plt.plot(x, found_in, label = 'Time Steps', color='red')
    plt.ylabel('Found in Time Steps')

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='cyan')
    plt.ylabel('Epsilon Change')

    plt.tight_layout()
    plt.show()
    '''
    return total_rewards, found_in, eps
    ####DRAWING # OF TIME STEPS VS EPISODES AND REWARD####


def q_learn_constant_depth(rewards, framework_graph, graph, q_table, start):
    #print('decay rate:', decay_rate)
    #print('epsilon goes to 0 in:', 1/decay_rate, ' episodes')

    global epsilon
    global decay_rate
    global max_q_depth

    total_rewards = []
    found_in = []
    eps = []
    
    for episode in range(max_episode):
        max_q_depth = 3
        print('max_q_depth:', max_q_depth)
            
        print('****EPISODE ', episode, ' ****')
        state = start
        found = False
        time_step = 1
        while(not found):# and time_step < 100):
            action = choose_action(graph, q_table, state)
            
            for le in range(len(graph[state][0])):
                #print('LE:', le)
                #print('length:', len(graph[state][0]))
                if(graph[state][0][le][1] == action):
                    new = graph[state][0][le][0]

            #print(state, '->', new)
            calculate_new_q_with_depth(rewards, q_table, action, state, new, graph)
            state = new
            
            if(framework_graph[state]['Faulty'] == True):
                found = True
                #print('FOUND!')
                print('Found in:', time_step, ' time steps')

            time_step += 1
                
        epsilon -= decay_rate
        #print(q_table)

        ####CALCULATING TOTAL REWARD AT THE END OF THE EPISODE####
        tot_reward = 0
        for i in range(len(q_table)):
            for j in range(len(q_table[0])):
                if(q_table[i][j] != 'NA'):
                    tot_reward += q_table[i][j]
        total_rewards.append(tot_reward)
        found_in.append(time_step)
        eps.append(epsilon)
        print('Total Reward:', tot_reward)
        ####CALCULATING TOTAL REWARD AT THE END OF THE EPISODE####


    ####DRAWING # OF TIME STEPS VS EPISODES AND REWARD####
    '''
    x = []
    for i in range(max_episode):
        x.append(i)
        
    plt.subplot(3,1,1)
    plt.plot(x, total_rewards, label = 'Total Reward', color='black')
    #plt.plot(x, found_in, label = 'Time Steps', color='red')
    plt.ylabel('Total Reward')

    plt.subplot(3,1,2)
    plt.plot(x, found_in, label = 'Time Steps', color='red')
    plt.ylabel('Found in Time Steps')

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='cyan')
    plt.ylabel('Epsilon Change')

    plt.tight_layout()
    plt.show()
    '''
    return total_rewards, found_in, eps
    ####DRAWING # OF TIME STEPS VS EPISODES AND REWARD####
        

def convert_to_list_representation(framework_graph):
    print(framework_graph[0])

    graph = [[[],[]] for i in range(len(framework_graph))]

    ##graph[i][0] -> State i outgoing edges
    ##graph[i][1] -> State i incoming edges
    for i in range(len(framework_graph)):
        for item in framework_graph[i]['Outgoing_edges']:
            graph[i][0].append((int(item[0][1:]),int(item[1])))
        for item in framework_graph[i]['Incoming_edges']:
            try:
                graph[i][1].append((int(item[0][1:]),int(item[1])))
            except:
                print('Fault:', item)

    ctr = -1
    for item in graph:
        ctr += 1
        print(ctr,item)

    return graph


def empty_q_table(graph, alphabet_card):
    ##SETTING UP THE Q_TABLE##
    q_table = [['NA' for i in range(alphabet_card)] for i in range(len(graph))]

    for i in range(len(graph)):
        for item in graph[i][0]:
            try:
                q_table[i][item[1]] = 0
            except:
                print(item[1])
    
    for item in q_table:
        print(item)
    ##SETTING UP THE Q_TABLE##
    return q_table

def get_rewards(framework_graph):
    ##SETTING UP THE REWARDS##
    rewards = []

    for i in range(len(framework_graph)):
        if(framework_graph[i]['Faulty'] == True):
            rewards.append(5)
        else:
            rewards.append(-1)
    ##SETTING UP THE REWARDS##

    return rewards

def main():
    
    ##READING THE GRAPH##
    framework_graph = graph_reader.return_graph('generating4/20_0.2_10_strong2_#1.csv')
    alphabet_card = 10
    ##READING THE GRAPH##

    ##CONVERTING GRAPH REPRESENTATION##
    graph = convert_to_list_representation(framework_graph)
    ##CONVERTING GRAPH REPRESENTATION##

    global epsilon
    global decay_rate

    epsilon = 0.01
    decay_rate = 0

    #####ONE EXPERIMENT RUNTIME SARSA#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    s_tot, s_found, eps = sarsa(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME SARSA#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    q_tot, q_found, eps = q_learn(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    c_tot, c_found, eps = q_learn_constant_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    v_tot, v_found, eps = q_learn_varying_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####

    x = []
    for i in range(max_episode):
        x.append(i)

        
    plt.subplot(3,1,1)
    plt.title('20_0.2_10_strong2_#1.csv, Epsilon Greedy 0.01')
    plt.plot(x, s_tot, label = 'Sarsa', color='black')
    plt.plot(x, q_tot, label = 'Q Learner', color='cyan')
    plt.plot(x, c_tot, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_tot, label = 'Q Varying Depth', color='red')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(x, s_found, label = 'Sarsa', color='black')
    plt.plot(x, q_found, label = 'Q Learner', color='cyan')
    plt.plot(x, c_found, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_found, label = 'Q Varying Depth', color='red')
    plt.ylabel('Found in Time Steps')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='red')
    plt.ylabel('Epsilon Change')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('20_0.2_10_strong2_#1.csv_Epsilon_Greedy_0.01.pdf')

    s_converge = 0
    q_converge = 0
    c_converge = 0
    v_converge = 0

    
    for i in range(0,175):
        s_converge += s_found[i]

    for i in range(0,175):
        q_converge += q_found[i]

    for i in range(0,175):
        c_converge += c_found[i]

    for i in range(0,175):
        v_converge = v_found[i]


    converges = [s_converge, q_converge, c_converge, v_converge]

    xx = ['SARSA', 'Q_LEARNER', 'CONSTANT DEPTH Q', 'VARYING DEPTH Q']
    collors = ['black', 'cyan', 'green', 'red']
    fig, ax = plt.subplots(1, 1)
    ax.bar(xx, converges, color=collors)
    plt.tight_layout()
    plt.title('Steps to Converge')
    plt.savefig('1.pdf')

    ######DECAY



    epsilon = 1
    decay_rate = 1/(max_episode-(max_episode/4))

    #####ONE EXPERIMENT RUNTIME SARSA#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    s_tot, s_found, eps = sarsa(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME SARSA#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    q_tot, q_found, eps = q_learn(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    c_tot, c_found, eps = q_learn_constant_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    v_tot, v_found, eps = q_learn_varying_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####

    x = []
    for i in range(max_episode):
        x.append(i)

        
    plt.subplot(3,1,1)
    plt.title('20_0.2_10_strong2_#1.csv, Decaying Epsilon')
    plt.plot(x, s_tot, label = 'Sarsa', color='black')
    plt.plot(x, q_tot, label = 'Q Learner', color='cyan')
    plt.plot(x, c_tot, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_tot, label = 'Q Varying Depth', color='red')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(x, s_found, label = 'Sarsa', color='black')
    plt.plot(x, q_found, label = 'Q Learner', color='cyan')
    plt.plot(x, c_found, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_found, label = 'Q Varying Depth', color='red')
    plt.ylabel('Found in Time Steps')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='red')
    plt.ylabel('Epsilon Change')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('20_0.2_10_strong2_#1.csv_Epsilon_Decaying.pdf')

    s_converge = max_episode
    q_converge = max_episode
    c_converge = max_episode
    v_converge = max_episode

    
    for i in range(5,max_episode):
        if(s_tot[i] == s_tot[i-1] and s_tot[i] == s_tot[i-2] and s_tot[i] == s_tot[i-3]):
            s_converge = i

    for i in range(5,max_episode):
        if(q_tot[i] == q_tot[i-1] and q_tot[i] == q_tot[i-2] and q_tot[i] == q_tot[i-3] ):
            q_converge = i

    for i in range(5,max_episode):
        if(c_tot[i] == c_tot[i-1] and c_tot[i] == c_tot[i-2] and c_tot[i] == c_tot[i-3] ):
            c_converge = i

    for i in range(5,max_episode):
        if(v_tot[i] == v_tot[i-1] and v_tot[i] == v_tot[i-2] and v_tot[i] == v_tot[i-3]):
            v_converge = i


    converges = [s_converge, q_converge, c_converge, v_converge]

    xx = ['SARSA', 'Q_LEARNER', 'CONSTANT DEPTH Q', 'VARYING DEPTH Q']
    collors = ['black', 'cyan', 'green', 'red']
    fig, ax = plt.subplots(1, 1)
    ax.bar(xx, converges, color=collors)
    plt.tight_layout()
    #plt.show()

    ####MEDIUM

    ##READING THE GRAPH##
    framework_graph = graph_reader.return_graph('generating4/200_0.2_15_strong2_#1.csv')
    alphabet_card = 15
    ##READING THE GRAPH##

    ##CONVERTING GRAPH REPRESENTATION##
    graph = convert_to_list_representation(framework_graph)
    ##CONVERTING GRAPH REPRESENTATION##

    
    epsilon = 0.01
    decay_rate = 0

    #####ONE EXPERIMENT RUNTIME SARSA#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    s_tot, s_found, eps = sarsa(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME SARSA#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    q_tot, q_found, eps = q_learn(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    c_tot, c_found, eps = q_learn_constant_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    v_tot, v_found, eps = q_learn_varying_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####

    x = []
    for i in range(max_episode):
        x.append(i)

        
    plt.subplot(3,1,1)
    plt.title('200_0.2_15_strong2_#1.csv, Epsilon Greedy 0.01')
    plt.plot(x, s_tot, label = 'Sarsa', color='black')
    plt.plot(x, q_tot, label = 'Q Learner', color='cyan')
    plt.plot(x, c_tot, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_tot, label = 'Q Varying Depth', color='red')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(x, s_found, label = 'Sarsa', color='black')
    plt.plot(x, q_found, label = 'Q Learner', color='cyan')
    plt.plot(x, c_found, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_found, label = 'Q Varying Depth', color='red')
    plt.ylabel('Found in Time Steps')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='red')
    plt.ylabel('Epsilon Change')
    plt.legend()

    plt.tight_layout()
    #plt.show
    plt.savefig('200_0.2_15_strong2_#1.csv_Epsilon_Greedy_0.01.pdf')

    s_converge = max_episode
    q_converge = max_episode
    c_converge = max_episode
    v_converge = max_episode

    
    for i in range(5,max_episode):
        if(s_tot[i] == s_tot[i-1] and s_tot[i] == s_tot[i-2] and s_tot[i] == s_tot[i-3] ):
            s_converge = i

    for i in range(5,max_episode):
        if(q_tot[i] == q_tot[i-1] and q_tot[i] == q_tot[i-2] and q_tot[i] == q_tot[i-3] ):
            q_converge = i

    for i in range(5,max_episode):
        if(c_tot[i] == c_tot[i-1] and c_tot[i] == c_tot[i-2] and c_tot[i] == c_tot[i-3] ):
            c_converge = i

    for i in range(5,max_episode):
        if(v_tot[i] == v_tot[i-1] and v_tot[i] == v_tot[i-2] and v_tot[i] == v_tot[i-3]):
            v_converge = i


    converges = [s_converge, q_converge, c_converge, v_converge]

    xx = ['SARSA', 'Q_LEARNER', 'CONSTANT DEPTH Q', 'VARYING DEPTH Q']
    collors = ['black', 'cyan', 'green', 'red']
    fig, ax = plt.subplots(1, 1)
    ax.bar(xx, converges, color=collors)
    plt.tight_layout()
    #plt.show

    ######DECAY



    epsilon = 1
    decay_rate = 1/(max_episode-(max_episode/4))

    #####ONE EXPERIMENT RUNTIME SARSA#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    s_tot, s_found, eps = sarsa(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME SARSA#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    q_tot, q_found, eps = q_learn(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    c_tot, c_found, eps = q_learn_constant_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    v_tot, v_found, eps = q_learn_varying_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####

    x = []
    for i in range(max_episode):
        x.append(i)

        
    plt.subplot(3,1,1)
    plt.title('200_0.2_15_strong2_#1.csv, Decaying Epsilon')
    plt.plot(x, s_tot, label = 'Sarsa', color='black')
    plt.plot(x, q_tot, label = 'Q Learner', color='cyan')
    plt.plot(x, c_tot, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_tot, label = 'Q Varying Depth', color='red')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(x, s_found, label = 'Sarsa', color='black')
    plt.plot(x, q_found, label = 'Q Learner', color='cyan')
    plt.plot(x, c_found, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_found, label = 'Q Varying Depth', color='red')
    plt.ylabel('Found in Time Steps')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='red')
    plt.ylabel('Epsilon Change')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('200_0.2_10_strong2_#1.csv_Decaying.pdf')

    s_converge = max_episode
    q_converge = max_episode
    c_converge = max_episode
    v_converge = max_episode

    
    for i in range(5,max_episode):
        if(s_tot[i] == s_tot[i-1] and s_tot[i] == s_tot[i-2] and s_tot[i] == s_tot[i-3]):
            s_converge = i

    for i in range(5,max_episode):
        if(q_tot[i] == q_tot[i-1] and q_tot[i] == q_tot[i-2] and q_tot[i] == q_tot[i-3] ):
            q_converge = i

    for i in range(5,max_episode):
        if(c_tot[i] == c_tot[i-1] and c_tot[i] == c_tot[i-2] and c_tot[i] == c_tot[i-3] ):
            c_converge = i

    for i in range(5,max_episode):
        if(v_tot[i] == v_tot[i-1] and v_tot[i] == v_tot[i-2] and v_tot[i] == v_tot[i-3]):
            v_converge = i


    converges = [s_converge, q_converge, c_converge, v_converge]

    xx = ['SARSA', 'Q_LEARNER', 'CONSTANT DEPTH Q', 'VARYING DEPTH Q']
    collors = ['black', 'cyan', 'green', 'red']
    fig, ax = plt.subplots(1, 1)
    ax.bar(xx, converges, color=collors)
    plt.tight_layout()
    #plt.show()


    ######BIG

    ##READING THE GRAPH##
    framework_graph = graph_reader.return_graph('generating4/2000_0.2_17_strong2_#1.csv')
    alphabet_card = 17
    ##READING THE GRAPH##

    ##CONVERTING GRAPH REPRESENTATION##
    graph = convert_to_list_representation(framework_graph)
    ##CONVERTING GRAPH REPRESENTATION##

    epsilon = 0.01
    decay_rate = 0

    #####ONE EXPERIMENT RUNTIME SARSA#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    s_tot, s_found, eps = sarsa(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME SARSA#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    q_tot, q_found, eps = q_learn(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    c_tot, c_found, eps = q_learn_constant_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    v_tot, v_found, eps = q_learn_varying_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####

    x = []
    for i in range(max_episode):
        x.append(i)

        
    plt.subplot(3,1,1)
    plt.title('2000_0.2_17_strong2_#1.csv, Epsilon Greedy 0.01')
    plt.plot(x, s_tot, label = 'Sarsa', color='black')
    plt.plot(x, q_tot, label = 'Q Learner', color='cyan')
    plt.plot(x, c_tot, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_tot, label = 'Q Varying Depth', color='red')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(x, s_found, label = 'Sarsa', color='black')
    plt.plot(x, q_found, label = 'Q Learner', color='cyan')
    plt.plot(x, c_found, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_found, label = 'Q Varying Depth', color='red')
    plt.ylabel('Found in Time Steps')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='red')
    plt.ylabel('Epsilon Change')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('2000_0.2_10_strong2_#1.csv_Epsilon_Greedy_0.01.pdf')

    s_converge = max_episode
    q_converge = max_episode
    c_converge = max_episode
    v_converge = max_episode

    
    for i in range(5,max_episode):
        if(s_tot[i] == s_tot[i-1] and s_tot[i] == s_tot[i-2] and s_tot[i] == s_tot[i-3] ):
            s_converge = i

    for i in range(5,max_episode):
        if(q_tot[i] == q_tot[i-1] and q_tot[i] == q_tot[i-2] and q_tot[i] == q_tot[i-3] ):
            q_converge = i

    for i in range(5,max_episode):
        if(c_tot[i] == c_tot[i-1] and c_tot[i] == c_tot[i-2] and c_tot[i] == c_tot[i-3] ):
            c_converge = i

    for i in range(5,max_episode):
        if(v_tot[i] == v_tot[i-1] and v_tot[i] == v_tot[i-2] and v_tot[i] == v_tot[i-3]):
            v_converge = i


    converges = [s_converge, q_converge, c_converge, v_converge]

    xx = ['SARSA', 'Q_LEARNER', 'CONSTANT DEPTH Q', 'VARYING DEPTH Q']
    collors = ['black', 'cyan', 'green', 'red']
    fig, ax = plt.subplots(1, 1)
    ax.bar(xx, converges, color=collors)
    plt.tight_layout()
    #plt.show()

    ######DECAY



    epsilon = 1
    decay_rate = 1/(max_episode-(max_episode/4))

    #####ONE EXPERIMENT RUNTIME SARSA#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    s_tot, s_found, eps = sarsa(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME SARSA#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    q_tot, q_found, eps = q_learn(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    c_tot, c_found, eps = q_learn_constant_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    v_tot, v_found, eps = q_learn_varying_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####

    x = []
    for i in range(max_episode):
        x.append(i)

        
    plt.subplot(3,1,1)
    plt.title('2000_0.2_17_strong2_#1.csv, Decaying Epsilon')
    plt.plot(x, s_tot, label = 'Sarsa', color='black')
    plt.plot(x, q_tot, label = 'Q Learner', color='cyan')
    plt.plot(x, c_tot, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_tot, label = 'Q Varying Depth', color='red')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(x, s_found, label = 'Sarsa', color='black')
    plt.plot(x, q_found, label = 'Q Learner', color='cyan')
    plt.plot(x, c_found, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_found, label = 'Q Varying Depth', color='red')
    plt.ylabel('Found in Time Steps')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='red')
    plt.ylabel('Epsilon Change')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('20_0.2_10_strong2_#1.csv_Decaying.pdf')

    s_converge = max_episode
    q_converge = max_episode
    c_converge = max_episode
    v_converge = max_episode

    
    for i in range(5,max_episode):
        if(s_tot[i] == s_tot[i-1] and s_tot[i] == s_tot[i-2] and s_tot[i] == s_tot[i-3]):
            s_converge = i

    for i in range(5,max_episode):
        if(q_tot[i] == q_tot[i-1] and q_tot[i] == q_tot[i-2] and q_tot[i] == q_tot[i-3] ):
            q_converge = i

    for i in range(5,max_episode):
        if(c_tot[i] == c_tot[i-1] and c_tot[i] == c_tot[i-2] and c_tot[i] == c_tot[i-3] ):
            c_converge = i

    for i in range(5,max_episode):
        if(v_tot[i] == v_tot[i-1] and v_tot[i] == v_tot[i-2] and v_tot[i] == v_tot[i-3]):
            v_converge = i


    converges = [s_converge, q_converge, c_converge, v_converge]

    xx = ['SARSA', 'Q_LEARNER', 'CONSTANT DEPTH Q', 'VARYING DEPTH Q']
    collors = ['black', 'cyan', 'green', 'red']
    fig, ax = plt.subplots(1, 1)
    ax.bar(xx, converges, color=collors)
    plt.tight_layout()
    #plt.show()


    #####BIGGEST

    ##READING THE GRAPH##
    framework_graph = graph_reader.return_graph('generating4/100000_0.2_20_strong2_#1.csv')
    alphabet_card = 20
    ##READING THE GRAPH##

    ##CONVERTING GRAPH REPRESENTATION##
    graph = convert_to_list_representation(framework_graph)
    ##CONVERTING GRAPH REPRESENTATION##

    epsilon = 0.01
    decay_rate = 0

    #####ONE EXPERIMENT RUNTIME SARSA#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    s_tot, s_found, eps = sarsa(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME SARSA#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    q_tot, q_found, eps = q_learn(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    c_tot, c_found, eps = q_learn_constant_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    v_tot, v_found, eps = q_learn_varying_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####

    x = []
    for i in range(max_episode):
        x.append(i)

        
    plt.subplot(3,1,1)
    plt.title('100000_0.2_20_strong2_#1.csv, Epsilon Greedy 0.01')
    plt.plot(x, s_tot, label = 'Sarsa', color='black')
    plt.plot(x, q_tot, label = 'Q Learner', color='cyan')
    plt.plot(x, c_tot, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_tot, label = 'Q Varying Depth', color='red')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(x, s_found, label = 'Sarsa', color='black')
    plt.plot(x, q_found, label = 'Q Learner', color='cyan')
    plt.plot(x, c_found, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_found, label = 'Q Varying Depth', color='red')
    plt.ylabel('Found in Time Steps')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='red')
    plt.ylabel('Epsilon Change')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('100000_0.2_10_strong2_#1.csv_Epsilon_Greedy_0.01.pdf')
    
    s_converge = max_episode
    q_converge = max_episode
    c_converge = max_episode
    v_converge = max_episode

    
    for i in range(5,max_episode):
        if(s_tot[i] == s_tot[i-1] and s_tot[i] == s_tot[i-2] and s_tot[i] == s_tot[i-3] ):
            s_converge = i

    for i in range(5,max_episode):
        if(q_tot[i] == q_tot[i-1] and q_tot[i] == q_tot[i-2] and q_tot[i] == q_tot[i-3] ):
            q_converge = i

    for i in range(5,max_episode):
        if(c_tot[i] == c_tot[i-1] and c_tot[i] == c_tot[i-2] and c_tot[i] == c_tot[i-3] ):
            c_converge = i

    for i in range(5,max_episode):
        if(v_tot[i] == v_tot[i-1] and v_tot[i] == v_tot[i-2] and v_tot[i] == v_tot[i-3]):
            v_converge = i


    converges = [s_converge, q_converge, c_converge, v_converge]

    xx = ['SARSA', 'Q_LEARNER', 'CONSTANT DEPTH Q', 'VARYING DEPTH Q']
    collors = ['black', 'cyan', 'green', 'red']
    fig, ax = plt.subplots(1, 1)
    ax.bar(xx, converges, color=collors)
    plt.tight_layout()
    #plt.show()

    ######DECAY



    epsilon = 1
    decay_rate = 1/(max_episode-(max_episode/4))

    #####ONE EXPERIMENT RUNTIME SARSA#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    s_tot, s_found, eps = sarsa(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME SARSA#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    q_tot, q_found, eps = q_learn(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    c_tot, c_found, eps = q_learn_constant_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN WITH CONSTANT DEPTH#####

    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####
    epsilon = 1
    q_table = empty_q_table(graph, alphabet_card)
    rewards = get_rewards(framework_graph)
    v_tot, v_found, eps = q_learn_varying_depth(rewards, framework_graph, graph, q_table, 10)
    #####ONE EXPERIMENT RUNTIME Q_LEARN VARYING DEPTH#####

    x = []
    for i in range(max_episode):
        x.append(i)

        
    plt.subplot(3,1,1)
    plt.title('100000_0.2_10_strong2_#1.csv, Decaying Epsilon')
    plt.plot(x, s_tot, label = 'Sarsa', color='black')
    plt.plot(x, q_tot, label = 'Q Learner', color='cyan')
    plt.plot(x, c_tot, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_tot, label = 'Q Varying Depth', color='red')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(x, s_found, label = 'Sarsa', color='black')
    plt.plot(x, q_found, label = 'Q Learner', color='cyan')
    plt.plot(x, c_found, label = 'Q Constant Depth', color='green')
    plt.plot(x, v_found, label = 'Q Varying Depth', color='red')
    plt.ylabel('Found in Time Steps')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='red')
    plt.ylabel('Epsilon Change')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('20_0.2_10_strong2_#1.csv_Epsilon_Decaying.pdf')

    s_converge = max_episode
    q_converge = max_episode
    c_converge = max_episode
    v_converge = max_episode

    
    for i in range(5,max_episode):
        if(s_tot[i] == s_tot[i-1] and s_tot[i] == s_tot[i-2] and s_tot[i] == s_tot[i-3]):
            s_converge = i

    for i in range(5,max_episode):
        if(q_tot[i] == q_tot[i-1] and q_tot[i] == q_tot[i-2] and q_tot[i] == q_tot[i-3] ):
            q_converge = i

    for i in range(5,max_episode):
        if(c_tot[i] == c_tot[i-1] and c_tot[i] == c_tot[i-2] and c_tot[i] == c_tot[i-3] ):
            c_converge = i

    for i in range(5,max_episode):
        if(v_tot[i] == v_tot[i-1] and v_tot[i] == v_tot[i-2] and v_tot[i] == v_tot[i-3]):
            v_converge = i


    converges = [s_converge, q_converge, c_converge, v_converge]

    xx = ['SARSA', 'Q_LEARNER', 'CONSTANT DEPTH Q', 'VARYING DEPTH Q']
    collors = ['black', 'cyan', 'green', 'red']
    fig, ax = plt.subplots(1, 1)
    ax.bar(xx, converges, color=collors)
    plt.tight_layout()
    #plt.show()


if __name__ == '__main__':
    random.seed(os.urandom(10000))
    main()

    
