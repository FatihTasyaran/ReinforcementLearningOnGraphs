import sys
import os
import random
from graphviz import Digraph
from graph_tool.all import *
import matplotlib.pyplot as plt
import math

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
                if (item[0] == in_node):
                    in_node = 'S'+str(node(no_states))
                    current_out = graph[int(out_node[1:])]['Outgoing_edges']
                    #print("Got that, new in:", in_node, 'current_out:', current_out, 'dont want:', in_node)
                    counter = counter + 1
                    #print('counter: ', counter)

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

    return graph, lib_graph, max_outer


def outer_frequency(graph, max_outer):
    x = []
    y = []

    for i in range(len(graph)):
        x.append(i)
        y.append(len(graph[i]['Outgoing_edges']))

    plt.hist(y, max_outer, histtype='stepfilled')
    plt.show()
    

def text_print(graph, no_states, density, card_alphabet):
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

    
def gt_draw(lib_graph):  
    print("Drawing graph")
    pos = arf_layout(lib_graph, max_iter=0)
    graph_draw(lib_graph, pos=pos , output="gt_4.pdf")

def weird_draw(bio_graph):

    print("Drawing bio graph")
    g = lib_graph
    g = GraphView(g, vfilt=label_largest_component(g))
    g.purge_vertices()
    state = minimize_nested_blockmodel_dl(g, deg_corr=True)
    t = get_hierarchy_tree(state)[0]
    tpos = pos = radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
    cts = get_hierarchy_control_points(g, t, tpos)
    pos = g.own_property(tpos)
    b = state.levels[0].b
    shape = b.copy()
    shape.a %= 14
    graph_draw(g, pos=pos, vertex_fill_color=b, vertex_shape=shape, edge_control_points=cts, edge_color=[0, 0, 0, 0.3], vertex_anchor=0, output="netscience_nested_mdl.pdf")

def gv_draw(graph):
    
    dot = Digraph(comment='graph_prop', engine='sfdp', node_attr={'width':'.0005', 'fixed_size':'True'})

    for item in graph:
        if(item['Faulty']):
            dot.node(item['State'], item['State'], color='red')
        else:
            dot.node(item['State'], item['State'])

        for edges in item['Outgoing_edges']:
            dot.edge(item['State'], edges[0], constraint = 'false', label= str(edges[1]))

    dot.render('test-output/round-table.gv', view=True)

def add_to_graph(graph, out_node, in_node, card_alphabet):
    out_state = 'S' + str(out_node)
    in_state = 'S' + str(in_node)

    actions = []
    for i in range(card_alphabet):
        actions.append(i)

    for i in range(len(graph[out_node]['Outgoing_edges'])):
        actions.remove(graph[out_node]['Outgoing_edges'][i][1])

    print("WILL CHOOSE FROM:" ,len(graph[out_node]['Outgoing_edges']), "WHERE MAX CARDINALITY IS:", card_alphabet)
    try:
        act = random.choice(actions)
        print('I chose:', act)
        graph[out_node]['Outgoing_edges'].append((in_state, act))
        graph[in_node]['Incoming_edges'].append((out_state, act))
        return 1
    except:
        print("State:", out_state, "Failed to find Action")
        return 0

        
        
    
        


##PRECONDITION: ONLY TO BE APPLIED TO RANDOM EDGES WHICH WILL NOT VIOLATE STRONGLY CONNECTED PROPERTY
##POSTCONDITION: WILL RETURN A STATE THAT OUT_NODE HAVE NO OUTGOING EDGE
def multi_transition_check(graph, out_node, in_node, no_states):
    good_returner = in_node

    outgoing_states = []

    for item in graph[out_node]['Outgoing_edges']:
        outgoing_states.append(int(item[0][1:]))
        
    print("OUTGOING_STATES: ", outgoing_states)

    exist = True
    while(exist):
        if good_returner in outgoing_states:
            good_returner = node(no_states)
            print("CHANGED: ", in_node, good_returner)
        else:
            exist = False

    print("I HAVE BEEN SUMMONED:", out_node, in_node, "RETURNED: ", good_returner)

    return good_returner

def dense_sanity_check(graph, out_node, in_node):
    outgoing_states = []

    for item in graph[out_node]['Outgoing_edges']:
        outgoing_states.append(int(item[0][1:]))

    if in_node in outgoing_states:
        return False
    else:
        return True
        
    

def generate_strongly_connected_graph(no_states, no_edges, density, card_alphabet):
    graph = [{} for i in range(no_states)]
    remaining_edges = no_edges

    for i in range(len(graph)):
        graph[i]['State'] = 'S'+str(i)
        graph[i]['Outgoing_edges'] = []
        graph[i]['Incoming_edges'] = []
        graph[i]['Faulty'] = False

    threshold = 2000
    no_clusters = 0

    if(no_states < threshold):
        no_clusters = math.log2(no_states)
    else:
        no_clusters = int(no_states/150)

    no_clusters = int(no_clusters)
    cluster_size = int(no_states/no_clusters)

    print('No clusters: ', no_clusters)
    print('Cluster Size: ', cluster_size)

    nodes = []
    nodes_remainings= []

    for i in range(0, no_states):
        nodes.append(i)
        nodes_remainings.append(i)


    clusters = [[] for i in range(no_clusters)]
    masters = []
    for c in range(0, no_clusters):

        for n in range(cluster_size):
            chosen = random.choice(nodes)
            clusters[c].append(chosen)
            nodes.remove(chosen)

    remaining = no_states-(cluster_size*no_clusters)
    if(remaining):
        for item in nodes:
            clusters[len(clusters)-1].append(item)

    for i in range(len(clusters)):
        print('###CLUSTER ', i , '###')
        print(clusters[i])

    if(no_edges < (2*no_states)):

        print("SPARSE ALGORITHM")
        
        for cluster in clusters:

            for i in range(len(cluster)-1):
                out_node = cluster[i]
                in_node = cluster[i+1]
                
                add_to_graph(graph, out_node, in_node, card_alphabet)
                remaining_edges = remaining_edges - 1

            out_node = cluster[len(cluster)-1]
            in_node = cluster[0]

            add_to_graph(graph, out_node, in_node, card_alphabet)
            remaining_edges = remaining_edges - 1

        first_master = 0
        for i in range(len(clusters)-1):
            
            out_node = random.choice(clusters[i])
            in_node = random.choice(clusters[i+1])
            if(i == 0):
                first_master = out_node

            add_to_graph(graph, out_node, in_node, card_alphabet)
            remaining_edges = remaining_edges - 1

        out_node = random.choice(clusters[len(clusters)-1])
        in_node = first_master

        add_to_graph(graph, out_node, in_node, card_alphabet)
        remaining_edges = remaining_edges - 1

        
        for i in range(remaining_edges):
            out_node = node(no_states)
            in_node = node(no_states)

            in_node = multi_transition_check(graph, out_node, in_node, no_states)

            add_to_graph(graph, out_node, in_node, card_alphabet)

    else:
        print("DENSE ALGORITHM")

        for cluster in clusters:
            master = random.choice(cluster)
            masters.append(master)

            
            ##ADDING RANDOM EDGES##
            for e in range(2 * len(cluster)):
                out_node = random.choice(cluster)
                in_node = random.choice(cluster)

                unsuccess = True

                while(unsuccess):
                    if(not dense_sanity_check(graph, out_node, in_node)):
                        in_node = random.choice(cluster)
                    else:
                        unsuccess = False
                    
                
                add_to_graph(graph, out_node, in_node, card_alphabet)
                remaining_edges = remaining_edges - 1
            ##ADDING RANDOM EDGES##

            
            ##MAKES SURE THESE MAKES A CLUSTER##
            for item in cluster:
                if(len(graph[item]['Outgoing_edges']) == 0 and len(graph[item]['Incoming_edges']) == 0):
                    out_node = random.choice(cluster)

                    while(len(graph[out_node]['Incoming_edges']) == 0):
                        out_node = random.choice(cluster)

                    in_node = item

                    add_to_graph(graph, out_node, in_node, card_alphabet)
                    remaining_edges = remaining_edges - 1
            ##MAKES SURE THESE MAKES A CLUSTER##
            

            ##CONNECTING MASTER TO ROOTS##
            for item in cluster:
                if(len(graph[item]['Outgoing_edges']) == 0):
                    out_node = item
                    in_node = master

                    add_to_graph(graph, out_node, in_node, card_alphabet)                    
                    remaining_edges = remaining_edges - 1

            ##CONNECTING MASTER TO ROOTS##

            ##CONNECTING LEAF TO MASTER##
                elif(len(graph[item]['Incoming_edges']) == 0):
                    out_node = master
                    in_node = item

                    add_to_graph(graph, out_node, in_node, card_alphabet)
                    remaining_edges = remaining_edges - 1
            ##CONNECTING LEAF TO MASTER##
            

        ##CONNECT MASTERS TO FORM A LOOP##
        for i in range(len(masters)-1):
            out_node = masters[i]
            in_node = masters[i+1]

            add_to_graph(graph, out_node, in_node, card_alphabet)                    
            remaining_edges = remaining_edges - 1

        out_node = masters[len(masters)-1]
        in_node = masters[0]

        add_to_graph(graph, out_node, in_node, card_alphabet)                
        remaining_edges = remaining_edges - 1
        ##CONNECT MASTERS TO FORM A LOOP##

            
        ##ADD REMAINING EDGES##    
        for i in range(remaining_edges):
            out_node = random.choice(nodes_remainings)
            in_node = random.choice(nodes_remainings)

            unsuccess = True

            while(unsuccess):
                if(not dense_sanity_check(graph, out_node, in_node)):
                    in_node = random.choice(cluster)
                else:
                    unsuccess = False
            
            add_to_graph(graph, out_node, in_node, card_alphabet)
            remaining_edges = remaining_edges - 1
        ##ADD REMAINING EDGES##
        
        print('remaining_edges:', remaining_edges)
        
    return graph
    

def main():
    random.seed(os.urandom(100000))

    no_states = int(sys.argv[1])
    density = float(sys.argv[2])
    card_alphabet = int(sys.argv[3])

    no_max_edges = no_states*card_alphabet
    no_edges = int(density*no_max_edges)
    print("No edges: ", no_edges)
    print("No max edges: ", no_states*(card_alphabet))

    if((card_alphabet* no_states * density) <= no_states):
        print('Not possible to generate Strongly Connecting DFA')
        print('Exiting...')
        exit()

    strong_graph = generate_strongly_connected_graph(no_states, no_edges, density, card_alphabet)
    gv_draw(strong_graph)
    text_print(strong_graph, no_states, density, card_alphabet)
    outer_frequency(strong_graph, 100)


    ####GENERATE WEAKLY CONNECTED GRAPH####
    #graph, lib_graph, max_outer = generate_weakly_connected_graph(no_states, no_edges, density, card_alphabet)
    #generate_strongly_connected_graph()
    #outer_frequency(graph, max_outer)
    #gt_draw(lib_graph)
    #gv_draw(graph)

    #prop_map, array = graph_tool.topology.label_components(lib_graph)
    #print(prop_map)
    #print(array)
    ####GENERATE WEAKLY CONNECTED GRAPH####
    
    

if __name__ == '__main__':
    main()
    

    




    

