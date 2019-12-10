import graph_reader
import random
import os

learning_rate = 0.75
discount_rate = 0.5
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
    
    val = random.randint(0, 20) / 20
    #print('val:', val)

    possible_actions = []
    for item in graph[state]['Outgoing_edges']:
        possible_actions.append(int(item[1]))

    if(val < epsilon):
        action = random.choice(possible_actions)
    else:
        val = -10
        for i in range(len(q_table[state])):
            if(q_table[state][i] != '-'):
                if(q_table[state][i] > val):
                    val = q_table[state][i]
                    action = i
        

    epsilon = epsilon - 0.01

    #print('State:', state, 'Possible Actions:', possible_actions)
    #print('Action:', action, 'Epsilon:', epsilon)

    return action


def q_learn(graph, card_alphabet, rewards, start_state):
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


    ##START LEARNING##
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
        
    ##START LEARNING##


    print('\n\n*******SECOND TRY*******\n\n')

    epsilon = 1
    
    ##SECOND TRY##
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
    ##SECOND TRY##


    print('\n\n*******THIRD TRY*******\n\n')

    epsilon = 1
    
    ##THIRD TRY##
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
    ##THIRD TRY##
            
        

        #print_q_table(q_table)


def main():
    graph = graph_reader.return_graph('10s/60_0.7_11_strong2_#4.csv')
    rewards = []

    for i in range(len(graph)):
        if(graph[i]['Faulty'] == True):
            rewards.append(10)
        else:
            rewards.append(-1)

    #print('Rewards:', rewards)
    
    ###
    
    ###
    card_alphabet = 11
    start = random.randint(0, len(graph))
    q_learn(graph, card_alphabet, rewards, start)
    #print(graph)

if __name__ == '__main__':
    random.seed(os.urandom(10000))
    main()






