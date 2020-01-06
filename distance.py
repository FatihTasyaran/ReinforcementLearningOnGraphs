import graph_reader
import networkx as nx




file1 = '20_0.2_10_strong2_#1.csv'
file2 = '20_0.7_10_strong2_#1.csv'
len1 = 20

file3 = '200_0.2_15_strong2_#1.csv'
file4 = '200_0.7_15_strong2_#1.csv'
len2 = 200

file5 = '2000_0.2_17_strong2_#1.csv'
file6 = '2000_0.7_17_strong2_#1.csv'
len3 = 2000

file7 = '100000_0.2_20_strong2_#1.csv'
file8 = '100000_0.7_20_strong2_#1.csv'
len4 = 2000



path = 'generating4/'
graph = graph_reader.return_graph(path+file1)
nx_graph = graph_reader.convert_to_networkX(graph)

my_max = 0
twos = -1
max_dist = -1
for i in range(len1):
    dist = nx.shortest_path_length(nx_graph,source=i,target=7)
    if (dist > my_max):
        my_max = i
        max_dist = dist
    if (dist == 2):
        twos = i

print(file1, ':  max: ', my_max, ' two: ', twos, ' max_dist: ', max_dist)


graph = graph_reader.return_graph(path+file2)
nx_graph = graph_reader.convert_to_networkX(graph)

my_max = 0
twos = -1
max_dist = -1
for i in range(len1):
    dist = nx.shortest_path_length(nx_graph,source=i,target=3)
    if (dist > my_max):
        my_max = i
        max_dist = dist
    if (dist == 2):
        twos = i

print(file2, ':  max: ', my_max, ' two: ', twos, ' max_dist: ', max_dist)
    


graph = graph_reader.return_graph(path+file3)
nx_graph = graph_reader.convert_to_networkX(graph)

my_max = 0
twos = -1
max_dist = -1
for i in range(len1):
    dist = nx.shortest_path_length(nx_graph,source=i,target=199)
    if (dist > my_max):
        my_max = i
        max_dist = dist
    if (dist == 2):
        twos = i

print(file3, ':  max: ', my_max, ' two: ', twos, ' max_dist: ', max_dist)


graph = graph_reader.return_graph(path+file4)
nx_graph = graph_reader.convert_to_networkX(graph)

my_max = 0
twos = -1
max_dist = -1
for i in range(len1):
    dist = nx.shortest_path_length(nx_graph,source=i,target=49)
    if (dist > my_max):
        my_max = i
        max_dist = dist
    if (dist == 2):
        twos = i
        
print(file4, ':  max: ', my_max, ' two: ', twos, ' max_dist: ', max_dist)


graph = graph_reader.return_graph(path+file5)
nx_graph = graph_reader.convert_to_networkX(graph)

my_max = 0
twos = -1
max_dist = -1
for i in range(len1):
    dist = nx.shortest_path_length(nx_graph,source=i,target=1674)
    if (dist > my_max):
        my_max = i
        max_dist = dist
    if (dist == 2):
        twos = i

print(file5, ':  max: ', my_max, ' two: ', twos, ' max_dist: ', max_dist)


graph = graph_reader.return_graph(path+file6)
nx_graph = graph_reader.convert_to_networkX(graph)

my_max = 0
twos = -1
max_dist = -1
for i in range(len1):
    dist = nx.shortest_path_length(nx_graph,source=i,target=822)
    if (dist > my_max):
        my_max = i
        max_dist = dist
    if (dist == 2):
        twos = i

print(file6, ':  max: ', my_max, ' two: ', twos, ' max_dist: ', max_dist)


graph = graph_reader.return_graph(path+file7)
nx_graph = graph_reader.convert_to_networkX(graph)

my_max = 0
twos = -1
max_dist = -1
for i in range(len1):
    dist = nx.shortest_path_length(nx_graph,source=i,target=32682)
    if (dist > my_max):
        my_max = i
        max_dist = dist
    if (dist == 2):
        twos = i

print(file7, ':  max: ', my_max, ' two: ', twos, ' max_dist: ', max_dist)


graph = graph_reader.return_graph(path+file8)
nx_graph = graph_reader.convert_to_networkX(graph)

my_max = 0
twos = -1
max_dist = -1
for i in range(len1):
    dist = nx.shortest_path_length(nx_graph,source=i,target=92124)
    if (dist > my_max):
        my_max = i
        max_dist = dist
    if (dist == 2):
        twos = i

print(file8, ':  max: ', my_max, ' two: ', twos, ' max_dist: ', max_dist)


