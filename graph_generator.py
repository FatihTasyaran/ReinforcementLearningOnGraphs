import sys
import os
import random
from graphviz import Digraph
from graph_tool.all import *
import matplotlib.pyplot as plt 

def node(no_states):
    return random.randint(0, no_states-1)

def action(card_alphabet):
    return random.randint(0, card_alphabet-1)

def generate_weakly_connected_graph(no_states, no_edges, density, card_alphabet):

    graph = [{} for i in range(no_states)]

    for i in range(len(graph)):
        graph[i]['State'] = 'S'+str(i)
        graph[i]['Outgoing_edges'] = []
        graph[i]['Faulty'] = False


        nodes = []

        for i in range(no_states):
            nodes.append('S'+str(i))

    connected = []
    unconnected = []

    for i in range(no_states):
        unconnected.append('S'+str(i))

    first = random.choice(unconnected)
    connected.append(first)
    unconnected.remove(first)
    while(len(unconnected) > 0):
        c = random.choice(connected)
        u = random.choice(unconnected)
        graph[int(c[1:])]['Outgoing_edges'].append((u,action(card_alphabet)))
        connected.append(u)
        unconnected.remove(u)


    for i in range(no_edges-no_states):   
        out_node = 'S'+str(node(no_states))
        in_node = 'S'+str(node(no_states))

        current_out = graph[int(out_node[1:])]['Outgoing_edges']
        repetition = True
        max_out = True

    
        while(max_out):
            if(len(current_out) > card_alphabet-1):
                #print('Changing node ', out_node, 'for reaching max outer degree')
                out_node = 'S'+str(node(no_states))
                current_out = graph[int(out_node[1:])]['Outgoing_edges']
            else:
                max_out = False
                #print('Node with less outer degree:', out_node)

        if(len(current_out) > card_alphabet-1):
            print('skipping')
            continue

        skipper = 0
        counter = 0

        while(repetition):
            for item in current_out:
                action
                if (item[0] == in_node):
                    in_node = 'S'+str(node(no_states))
                    current_out = graph[int(out_node[1:])]['Outgoing_edges']
                    #print("Got that, new in:", in_node, 'current_out:', current_out, 'dont want:', in_node)
                    counter = counter + 1
                    print('counter: ', counter)

                    if(counter > (no_states/2)):
                        skipper = 1
                        repetition = False

            repetition = False

        if(skipper):
            print('skipped this iteration')
            continue
    
        try:
            graph[int(out_node[1:])]['Outgoing_edges'].append((in_node,action(card_alphabet)))
        except:
            print("ERROR:", out_node)

    no_faulty = no_states*0.05
    no_faulty = int(no_faulty)
    if (no_faulty == 0):
        no_faulty = 1

    for i in range(no_faulty):
        fault_loc = node(no_states)
        graph[fault_loc]['Faulty'] = True
        
    for item in graph:
        print("###", item['State'], "###")
        print(item, '\n')

    graph_prop = '# of Nodes: ' + str(no_states) + ' Density: ' + str(density) + ' Alphabet Cardinality: ' + str(card_alphabet)
    print(graph_prop)

    max_outer = 0
    loc = 0
    for i in range(len(graph)):
        if(len(graph[i]['Outgoing_edges']) > max_outer):
            #print('Max outer changed to: ', len(graph[i]['Outgoing_edges']), graph[i]['Outgoing_edges'])
            max_outer = len(graph[i]['Outgoing_edges'])
            loc = i

    print('Max outer Degree: ', max_outer, ' which is: ', graph[loc])

    lib_graph = Graph() ##A directed graph


    for item in graph:
        item['lib_node'] = lib_graph.add_vertex()
    
    for item in graph:
        item['lib_edge'] = []
        for edge in item['Outgoing_edges']:
            #print(edge)
            int_node = int(edge[0][1:])
            lib_graph.add_edge(item['lib_node'], graph[int_node]['lib_node'])

    #print("Calculate positioning")
    #pos = arf_layout(lib_graph)


    ###DRAWING OUTER EDGE FREQUENCY###
    x = []
    y = []

    for i in range(len(graph)):
        x.append(i)
        y.append(len(graph[i]['Outgoing_edges']))

    #y.sort(reverse=True)
    #plt.plot(x, y, linewidth=.2)
    plt.hist(y, max_outer, histtype='stepfilled')
    plt.show()

    ###DRAWING OUTER EDGE FREQUENCY###


    print_graph = 1
    if(print_graph):
        print("Drawing graph")
        graph_draw(lib_graph, output_size=(300,300), output="gt_4.pdf")



def main():
    random.seed(os.urandom(100000))

    no_states = int(sys.argv[1])
    density = float(sys.argv[2])
    card_alphabet = int(sys.argv[3])

    no_max_edges = no_states*card_alphabet
    no_edges = int(density*no_max_edges)
    print("No edges: ", no_edges)
    print("No max edges: ", no_states*(card_alphabet))
    
    generate_weakly_connected_graph(no_states, no_edges, density ,card_alphabet)
    #generate_strongly_connected_graph()
    #gt_draw()
    #gv_draw()
    #draw_outer_degrees()

if __name__ == '__main__':
    main()
    
'''
dot = Digraph(comment=graph_prop, engine='sfdp', node_attr={'width':'.0005', 'fixed_size':'True'})

for item in graph:
    if(item['Faulty']):
        dot.node(item['State'], item['State'], color='red')
    else:
        dot.node(item['State'], item['State'])

    for edges in item['Outgoing_edges']:
        dot.edge(item['State'], edges[0], constraint = 'false', label= str(edges[1]))

dot.render('test-output/round-table.gv', view=True)
'''
    




    

