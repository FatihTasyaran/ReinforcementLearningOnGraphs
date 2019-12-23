import graph_reader
import random
import os
import matplotlib.pyplot as plt
import time
import operator

learning_rate = 0.95
discount_rate = 0.75
epsilon = 1

def calculate_new_q(graph, state, action, q_table, rewards):
    #print('q at calculating:', q_table, 'a:', action)
    try:
        old_q = q_table[state][action]
    except:
        print(len(q_table[state]), action)
        #print('old_q:', old_q)

    ####NEW STATE####
    old_state = graph[state]

    possibru = graph[state]['Outgoing_edges']
    #print('Possbru:', possibru)
    new_state = 0

    global_max = -1

    for item in possibru:
        if(rewards[int(item[0][1:])] > global_max):
            global_max = rewards[int(item[0][1:])]
        if(action == item[1]):
            new_state = int(item[0][1:])
    ####NEW STATE####

    
    R_t1 = rewards[new_state]
    new_q = (1-learning_rate) * old_q + learning_rate * (R_t1 + discount_rate * global_max)

    q_table[state][action] = new_q
    


def print_q_table(q_table):
    for i in range(len(q_table)):
        print(q_table[i])

def choose_action(graph, state, q_table, rewards):

    global epsilon
    
    val = random.randint(0, 1000)
    #print('val:', val)

    possible_actions_list = []
    possible_actions = {}
    for item in graph[state]['Outgoing_edges']:
        possible_actions_list.append(int(item[1]))
        possible_actions[str(item[1])] = q_table[state][int(item[1])]
    print(possible_actions)

    if(val > epsilon):
        print('Random Choice')
        action = random.choice(possible_actions_list)

    else:
        #action = int(max(possible_actions, key=int))
        action = int(max(possible_actions.items(), key=operator.itemgetter(1))[0])
        print('Best among them: ', action)
        #print('Best Among:', max(possible_actions, key=int))
    '''
    else:
        val = -100
        for i in range(len(q_table[state])):
            if(q_table[state][i] != '-'):
                if(q_table[state][i] >= val):
                    val = q_table[state][i]
                    action = i
     '''

    epsilon = epsilon - 0.001

    #print('State:', state, 'Possible Actions:', possible_actions)
    #print('Action:', action, 'Epsilon:', epsilon)

    return action


def q_learn(graph, card_alphabet, rewards, start_state, filename):

    global epsilon
    
    original_state = start_state
    q_table = [[] for i in range (len(graph))]

    for item in q_table:
        for i in range(card_alphabet):
            item.append(0)

    for node in range (len(graph)):
        actions = []

        for i in range(card_alphabet):
            actions.append(i)

        for edge in graph[node]['Outgoing_edges']:
            try:
                actions.remove(int(edge[1]))
            except:
                print(graph[node])

        for item in actions:
            q_table[node][item] = '-'


    epoch = 10000

    x = []
    total_at_end = []
    found_in = []

    for i in range(epoch):
        x.append(i)
            
    for epoc in range(epoch):    
         print("######EPOCH ", epoc, " ########")
         epsilon = epoc
         start_state = original_state
         found = False
         episodes = 0

         while(not found and episodes < 100):

             action = choose_action(graph, start_state, q_table, rewards)
             calculate_new_q(graph, start_state, action, q_table, rewards)

             total_reward = 0
             for i in range(len(q_table)):
                 for j in range(len(q_table[0])):
                     if(q_table[i][j] != '-'):
                         total_reward += q_table[i][j]

             print('**Episode:', episodes, '**', 'State', start_state, '->', action)
             print('Total Reward:', total_reward)


             for item in graph[start_state]['Outgoing_edges']:
                 if(int(item[1]) == action):
                     start_state = int(item[0][1:])

             episodes += 1

             
             if(graph[start_state]['Faulty'] == True):
                print('Found in', episodes, ' episodes')
                found = True
                total_at_end.append(total_reward)
                found_in.append(episodes)
                
             elif(episodes == 99 and found == False):
                total_at_end.append(total_reward)
                found_in.append(episodes)
                
            #time.sleep(1)
            
                
                
    ##START LEARNING##
    plt.subplot(2,1,1)
    plt.plot(x, total_at_end, label = 'Total Reward', color='black')
    plt.ylabel('Total Reward Over Epochs')
    plt.title(filename + " " + str(epoch) + " epochs")

    plt.subplot(2,1,2)
    plt.plot(x, found_in, label = 'Episodes', color='red')
    plt.ylabel('Found in Episodes')
    
    plt.show()

         


def main():
    graph = graph_reader.return_graph('10s/15_0.7_11_strong2_#4.csv')
    graph = graph_reader.return_graph('10s/60_0.7_11_strong2_#4.csv')
    
    
    #graph = graph_reader.return_graph('/home/fatih/Documents/CS560/project/generating2/700_0.2_10_strong2_#5.csv')
    rewards = []

    for i in range(len(graph)):
        if(graph[i]['Faulty'] == True):
            rewards.append(100)
        else:
            rewards.append(-2)

    #print('Rewards:', rewards)
    
    ###
    
    ###
    card_alphabet = 11
    #start = random.randint(0, len(graph))
    start = 45
    q_learn(graph, card_alphabet, rewards, start, '10s/15_0.7_11_strong2_#4.csv')
    #print(graph)

if __name__ == '__main__':
    random.seed(os.urandom(10000))
    main()
