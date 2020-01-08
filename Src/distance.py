import graph_reader
import networkx as nx

pathS = '/home/fatih/Documents/CS560/project/Data/10s/55_0.3_8_strong2_#4.csv'
pathB = '/home/fatih/Documents/CS560/project/Data/generating4/2000_0.2_17_strong2_#1.csv'

graphS = graph_reader.return_graph(pathS)
nx_graphS = graph_reader.convert_to_networkX(graphS)


graphB = graph_reader.return_graph(pathB)
nx_graphB = graph_reader.convert_to_networkX(graphB)

print('Small')
print(nx.shortest_path_length(nx_graphS,source=0,target=1))
print(nx.shortest_path_length(nx_graphS,source=1,target=2))
print(nx.shortest_path_length(nx_graphS,source=2,target=32))

print('Big')
print(nx.shortest_path_length(nx_graphB,source=0,target=1))
print(nx.shortest_path_length(nx_graphB,source=1,target=2))
print(nx.shortest_path_length(nx_graphB,source=2,target=1674))



