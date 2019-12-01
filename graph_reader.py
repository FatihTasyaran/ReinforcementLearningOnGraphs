import sys
from graphviz import Digraph
from graph_tool.all import *
import matplotlib.pyplot as plt
import math
import networkx as nx

def read_from_file(filename):
    rdr = open(filename, 'r')
    whole = rdr.readlines()

    graph = [{} for i in range(len(whole))]

    for i in range(len(graph)):
        graph[i]['State'] = 'DUMMY'
        graph[i]['Outgoing_edges'] = []
        graph[i]['Incoming_edges'] = []
        graph[i]['Faulty'] = False

    for l in range(len(whole)):
        line = whole[l]
        print('line:', line)
        info = line.split(',')

        graph[l]['State'] = info[0]
        out_edges_finish = 0

        for i in range(len(info)):
            if(info[i] == 'In'):
                out_edges_finish = i

        for i in range(2, out_edges_finish, 2):
            graph[l]['Outgoing_edges'].append((info[i],info[i+1]))

        for i in range(out_edges_finish+1, len(info)-2, 2):
            graph[l]['Incoming_edges'].append((info[i],info[i+1]))

        if(info[len(info)-1] == 'True\n'):
            graph[l]['Faulty'] = True
        

        
        
    return graph

def convert_to_graphtools(graph):
    gt_graph = Graph() ##A directed graph

    for item in graph:
        item['lib_node'] = gt_graph.add_vertex()

    for item in graph:
        item['lib_edge'] = []

        for edge in item['Outgoing_edges']:
            int_node = int(edge[0][1:])
            gt_graph.add_edge(item['lib_node'], graph[int_node]['lib_node'])

    return gt_graph

def convert_to_networkX(graph):
    nxg = nx.DiGraph()

    for item in graph:
        nxg.add_node(int(item['State'][1:]))

    for item in graph:

        for edge in item:
            edges = item['Outgoing_edges']
            for conn in edges:
                nxg.add_edge(int(item['State'][1:]),int(conn[0][1:]), action=conn[1])

    return nxg
    
    
#def convert_to_networkX():
def gt_draw(gt_graph):
    print("Drawing with graphtools")
    pos = arf_layout(gt_graph, max_iter=0)
    graph_draw(gt_graph, pos=pos, output="tryer.pdf")
    
#def dot_draw():
#def outer_degree_hist()
#def terminal_print():
#def validate():

def main():
    filename = sys.argv[1]
    graph = read_from_file(filename)
    #gt_graph = convert_to_graphtools(graph)
    #gt_draw(gt_graph)
    #checked = extract_largest_component(gt_graph, directed=True, prune=True)
    #gt_draw(checked)

    nx_graph = convert_to_networkX(graph)

    print('#Nodes:', len(nx_graph), 'Is strongly connected:', nx.is_strongly_connected(nx_graph))
    


if __name__ == '__main__':
    main()
