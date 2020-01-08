import graph_reader
import random
import os
import time
import matplotlib.pyplot as plt
import numpy as np

max_episode = 100
learning_rate = 0.9
discount = 0.75
epsilon = 1##Start with exploration
decay_rate = 1/(max_episode-(max_episode/5)) ##First half of episodes favor exploration,
                               ##Second half favor exploitation
max_q_depth = 1
CHECKPOINT1_FOUND = False
CHECKPOINT2_FOUND = False
CHECKPOINT1 = 1
CHECKPOINT2 = 2

#CHECKPOINT1 = 22
#CHECKPOINT2 = 5


def calculate_new_q(rewards1, rewards2, rewards3 ,q_table, action, old_state, new_state):
    ##IMPLEMENTING BELLMAN OPTIMALITY
    old = (1-learning_rate)*q_table[old_state][action]

    max_q = -1000 ##Cardinal value to be changed

    for i in range(len(q_table[new_state])):
        if(q_table[new_state][i] != 'NA'):
            if(q_table[new_state][i] > max_q):
                max_q = q_table[new_state][i]

    rewards = rewards1

    if(CHECKPOINT1_FOUND and not CHECKPOINT2_FOUND):
        rewards = rewards2

    if(CHECKPOINT1_FOUND and CHECKPOINT2_FOUND):
        rewards = rewards3

    
    
    #print('len rewards:', rewards)
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

def q_learn(rewards1, rewards2, rewards3, framework_graph, graph, q_table, start):
    #print('decay rate:', decay_rate)
    #print('epsilon goes to 0 in:', 1/decay_rate, ' episodes')

    global epsilon
    global decay_rate
    global CHECKPOINT1_FOUND
    global CHECKPOINT2_FOUND

    total_rewards = []
    found_in = []
    checkp1 = []
    checkp2 = []
    eps = []
    learned_min = []
    
    for episode in range(max_episode):
        print('****EPISODE ', episode, ' ****')
        state = start
        found = False
        time_step = 1
        #CHECKPOINT_FOUND = False
        while(not found):
            action = choose_action(graph, q_table, state)
            
            for le in range(len(graph[state][0])):
                #print('LE:', le)
                #print('length:', len(graph[state][0]))
                if(graph[state][0][le][1] == action):
                    new = graph[state][0][le][0]

            #print(state, '->', new)
            calculate_new_q(rewards1, rewards2, rewards3, q_table, action, state, new)
            state = new

            if(state == CHECKPOINT1):
                CHECKPOINT1_FOUND = True
                found = True
                print('Checkpoint Found in: ', time_step, ' steps')
                checkp1.append(time_step)

            time_step += 1

            if(CHECKPOINT1_FOUND):
                end = False
                while(not end):
                    action = choose_action(graph, q_table, state)
                    
                    for le in range(len(graph[state][0])):
                        if(graph[state][0][le][1] == action):
                            new = graph[state][0][le][0]

                    #print(state, '->', new)
                    calculate_new_q(rewards1, rewards2, rewards3, q_table, action, state, new)
                    state = new

                    if(state == CHECKPOINT2):
                        CHECKPOINT2_FOUND = True
                        end = True
                        print('Checkpoint Found in: ', time_step, ' steps')
                        checkp2.append(time_step)

                    time_step += 1

                    if(CHECKPOINT2_FOUND):
                        last = False

                        while(not last):

                            action = choose_action(graph, q_table, state)
                    
                            for le in range(len(graph[state][0])):
                                if(graph[state][0][le][1] == action):
                                    new = graph[state][0][le][0]

                            #print(state, '->', new)
                            calculate_new_q(rewards1, rewards2, rewards3, q_table, action, state, new)
                            state = new
            
                            if(framework_graph[state]['Faulty'] == True):
                                last = True
                                print('Found in:', time_step, ' time steps')

                            time_step += 1
                        
                    
                
        epsilon -= decay_rate
        CHECKPOINT1_FOUND = False
        CHECKPOINT2_FOUND = False
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
    x = []
    for i in range(max_episode):
        x.append(i)

    learned_min.append(checkp1[max_episode-1])
    learned_min.append(checkp2[max_episode-1])
    learned_min.append(found_in[max_episode-1])

    eps = 'null'
    if(decay_rate == 0):
        eps = str(epsilon) + ' Epsilon Greedy'
    else:
        eps = 'Epsilon Decaying'
    
    tit = str(len(framework_graph))
    title = tit + ' nodes ' + eps

    
        
    plt.subplot(3,1,1)
    plt.title(title)
    plt.plot(x, total_rewards, label = 'Total Reward', color='black')
    #plt.plot(x, found_in, label = 'Time Steps', color='red')
    plt.ylabel('Total Reward')

    plt.subplot(3,1,2)
    plt.plot(x, checkp1, label = 'Check Point1', color='green')
    plt.plot(x, checkp2, label = 'Check Point2', color='purple')
    plt.plot(x, found_in, label = 'Fault', color='red')
    plt.ylabel('Found in Time Steps')
    plt.legend()

    tot_c1 = sum(checkp1)
    tot_c2 = sum(checkp2)
    tot_f = sum(found_in)

    totals = [tot_c1, tot_c2, tot_f]
    xx = ['CHECKPOINT1', 'CHECKPOINT2', 'FAULTY']
    collors = ['green', 'purple', 'red']

    plt.subplot(3,1,3)
    plt.title('Total Time Steps Spent')
    plt.bar(xx, totals, color=collors)

    '''
    plt.subplot(3,1,3)
    plt.plot(x, eps, label = 'Epsilon', color='cyan')
    plt.ylabel('Epsilon Change')
    '''

    plt.tight_layout()
    #plt.show()
    name = str(epsilon) + '_' + str(max_episode) + 'big.pdf'
    plt.savefig(name)
    
    return total_rewards, found_in, eps, learned_min
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
    return total_rewards, found_in, eps
    ####DRAWING # OF TIME STEPS VS EPISODES AND REWARD####


def q_lern_constant_depth(rewards, framework_graph, graph, q_table, start):
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
    global CHECKPOINT
    rewards1 = []
    rewards2 = []
    rewards3 = []

    for i in range(len(framework_graph)):
        if(i == CHECKPOINT1):
            rewards1.append(10)
        else:
            rewards1.append(-2)

    for i in range(len(framework_graph)):
        if(i == CHECKPOINT2):
            rewards2.append(10)
        else:
            rewards2.append(-2)

    for i in range(len(framework_graph)):
        if(framework_graph[i]['Faulty'] == True):
            rewards3.append(10)
        else:
            rewards3.append(-2)
    ##SETTING UP THE REWARDS##

    return rewards1, rewards2, rewards3

def main():
    
    ##READING THE GRAPH##
    framework_graph = graph_reader.return_graph('/home/fatih/Documents/CS560/project/Data/10s/55_0.3_8_strong2_#4.csv')
    #framework_graph = graph_reader.return_graph('generating2/46000_0.2_25_strong2_#2.csv')
    #framework_graph = graph_reader.return_graph('generating3/180000_0.2_25_strong2_#2.csv')
    alphabet_card = 8
    ##READING THE GRAPH##

    ##CONVERTING GRAPH REPRESENTATION##
    graph = convert_to_list_representation(framework_graph)
    ##CONVERTING GRAPH REPRESENTATION##

    global epsilon
    global decay_rate
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####

    epsilon = 1
    decay_rate = 1/(max_episode-(max_episode/5)) 
    q_table = empty_q_table(graph, alphabet_card)
    rewards1, rewards2, rewards3 = get_rewards(framework_graph)
    q_tot, q_found, eps, one_learned_min = q_learn(rewards1, rewards2, rewards3, framework_graph, graph, q_table, 0)

    #####ONE EXPERIMENT RUNTIME Q_LEARN####
    
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 0.1
    decay_rate = 0
    q_table = empty_q_table(graph, alphabet_card)
    rewards1, rewards2, rewards3 = get_rewards(framework_graph)
    q_tot, q_found, eps, onef_learned_min = q_learn(rewards1, rewards2, rewards3, framework_graph, graph, q_table, 0)
    #####ONE EXPERIMENT RUNTIME Q_LEARN####
   
    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 0.01
    decay_rate = 0
    q_table = empty_q_table(graph, alphabet_card)
    rewards1, rewards2, rewards3 = get_rewards(framework_graph)
    q_tot, q_found, eps, twof_learned_min= q_learn(rewards1, rewards2, rewards3, framework_graph, graph, q_table, 0)
    #####ONE EXPERIMENT RUNTIME Q_LEARN####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 0.001
    decay_rate = 0
    q_table = empty_q_table(graph, alphabet_card)
    rewards1, rewards2, rewards3 = get_rewards(framework_graph)
    q_tot, q_found, eps , threef_learned_min= q_learn(rewards1, rewards2, rewards3, framework_graph, graph, q_table, 0)
    #####ONE EXPERIMENT RUNTIME Q_LEARN####

    collors = ['black', 'purple', 'cyan', 'red', 'green']

    real_small = [2,5,7]

    

    plt.subplot(3,1,1)
    titr = str(max_episode) + ' Episodes 55 Nodes Learned Min Distances \n Start to CHECKPOINT 1'
    plt.title(titr)
    xx = ['Real','Decaying', '0.1', '0.01', '0.001']
    vals = [real_small[0], one_learned_min[0], onef_learned_min[0], twof_learned_min[0], threef_learned_min[0]]
    plt.bar(xx, vals, color=collors)
    


    plt.subplot(3,1,2)
    plt.title('CHECKPOINT 1 to CHECKPOINT 2')
    xx = ['Real','Decaying', '0.1', '0.01', '0.001']
    vals = [real_small[1], one_learned_min[1], onef_learned_min[1], twof_learned_min[1], threef_learned_min[1]]
    plt.bar(xx, vals, color=collors)
    


    plt.subplot(3,1,3)
    plt.title('CHECKPOINT 2 to FAULTY')
    xx = ['Real','Decaying', '0.1', '0.01', '0.001']
    vals = [real_small[1], one_learned_min[1], onef_learned_min[1], twof_learned_min[1], threef_learned_min[1]]
    plt.bar(xx, vals, color=collors)

    plt.tight_layout()
    #plt.show()

    plt.savefig('100_small_learned_min.pdf')
    
    
    


    ####BIGGER#####
    framework_graph = graph_reader.return_graph('/home/fatih/Documents/CS560/proje\
ct/Data/generating4/2000_0.2_17_strong2_#1.csv')
    alphabet_card = 25
    graph = convert_to_list_representation(framework_graph)
    ####BIGGER#####


    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 1
    decay_rate = 1/(max_episode-(max_episode/5)) 
    q_table = empty_q_table(graph, alphabet_card)
    rewards1, rewards2, rewards3 = get_rewards(framework_graph)
    q_tot, q_found, eps, one_learned_min= q_learn(rewards1, rewards2, rewards3, framework_graph, graph, q_table, 0)
    #####ONE EXPERIMENT RUNTIME Q_LEARN####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 0.1
    decay_rate = 0
    q_table = empty_q_table(graph, alphabet_card)
    rewards1, rewards2, rewards3 = get_rewards(framework_graph)
    q_tot, q_found, eps, onef_learned_min = q_learn(rewards1, rewards2, rewards3, framework_graph, graph, q_table, 0)
    #####ONE EXPERIMENT RUNTIME Q_LEARN####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 0.01
    decay_rate = 0
    q_table = empty_q_table(graph, alphabet_card)
    rewards1, rewards2, rewards3 = get_rewards(framework_graph)
    q_tot, q_found, eps, twof_learned_min = q_learn(rewards1, rewards2, rewards3, framework_graph, graph, q_table, 0)
    #####ONE EXPERIMENT RUNTIME Q_LEARN####

    #####ONE EXPERIMENT RUNTIME Q_LEARN#####
    epsilon = 0.001
    decay_rate = 0
    q_table = empty_q_table(graph, alphabet_card)
    rewards1, rewards2, rewards3 = get_rewards(framework_graph)
    q_tot, q_found, eps, threef_learned_min = q_learn(rewards1, rewards2, rewards3, framework_graph, graph, q_table, 0)
    #####ONE EXPERIMENT RUNTIME Q_LEARN####

    collors = ['black', 'purple', 'cyan', 'red', 'green']

    real_small = [3,5,8]

    

    plt.subplot(3,1,1)
    titr = str(max_episode) + ' Episodes 2000 Nodes Learned Min Distances \n Start to CHECKPOINT 1'
    plt.title(titr)
    xx = ['Real','Decaying', '0.1', '0.01', '0.001']
    vals = [real_small[0], one_learned_min[0], onef_learned_min[0], twof_learned_min[0], threef_learned_min[0]]
    plt.bar(xx, vals, color=collors)
    


    plt.subplot(3,1,2)
    plt.title('CHECKPOINT 1 to CHECKPOINT 2')
    xx = ['Real','Decaying', '0.1', '0.01', '0.001']
    vals = [real_small[1], one_learned_min[1], onef_learned_min[1], twof_learned_min[1], threef_learned_min[1]]
    plt.bar(xx, vals, color=collors)
    


    plt.subplot(3,1,3)
    plt.title('CHECKPOINT 2 to FAULTY')
    xx = ['Real','Decaying', '0.1', '0.01', '0.001']
    vals = [real_small[1], one_learned_min[1], onef_learned_min[1], twof_learned_min[1], threef_learned_min[1]]
    plt.bar(xx, vals, color=collors)

    plt.tight_layout()
    #plt.show()
    plt.savefig('100_big_learned_min.pdf')


if __name__ == '__main__':
    random.seed(os.urandom(10000))
    main()

    
