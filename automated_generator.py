import graph_generator
import sys
import os


def main():
    least_nodes = int(sys.argv[1])
    most_nodes = int(sys.argv[2])
    density = float(sys.argv[3])
    card_alphabet = int(sys.argv[4])
    leap = int(sys.argv[5])
    algorithm = sys.argv[6]

    no_max_edges = least_nodes*card_alphabet
    no_edges = int(density*no_max_edges)

    nodes = []

    for i in range(least_nodes, most_nodes+least_nodes, leap):
        nodes.append(i)

    print(nodes)

    no = ''


    for node in nodes:
        
        for i in range(5):

            no = str(i+1)
            graph_generator.main(node, density , card_alphabet, algorithm, no)
            print("DONE")
            os.wait
    
        

if __name__ == '__main__':
    main()
