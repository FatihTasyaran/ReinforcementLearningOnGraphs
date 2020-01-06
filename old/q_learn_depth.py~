import graph_reader
import random
import os
import time
import matplotlib.pyplot as plt

max_episode = 500
learning_rate = 0.9
discount = 0.9
epsilon = 0.001 ##Start with exploration
decay_rate = 0#1/(max_episode) ##First half of episodes favor exploration,
                               ##Second half favor exploitation

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

            print(state, '->', new)
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
    
    plt.show()
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
        
    

def main():
    ##READING THE GRAPH##
    #framework_graph = graph_reader.return_graph('10s/55_0.3_8_strong2_#4.csv')
    #framework_graph = graph_reader.return_graph('generating2/96000_0.2_25_strong2_#2.csv')
    framework_graph = graph_reader.return_graph('generating3/180000_0.2_25_strong2_#2.csv')
    alphabet_card = 25
    ##READING THE GRAPH##


    
    ##CONVERTING GRAPH REPRESENTATION##
    graph = convert_to_list_representation(framework_graph)
    ##CONVERTING GRAPH REPRESENTATION##



    ##SETTING UP THE REWARDS##
    rewards = []

    for i in range(len(framework_graph)):
        if(framework_graph[i]['Faulty'] == True):
            rewards.append(5)
        else:
            rewards.append(-1)
    ##SETTING UP THE REWARDS##




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



    ##START ALGORITHM##
    q_learn(rewards, framework_graph, graph, q_table, 0)
    ##START ALGORITHM##


if __name__ == '__main__':
    random.seed(os.urandom(10000))
    main()

    
