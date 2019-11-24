import sys

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

        if(info[len(info)-1] == 'True'):
            graph[l]['Faulty'] = True

        
        
    return graph



def main():
    filename = sys.argv[1]
    graph = read_from_file(filename)
    print(graph)



if __name__ == '__main__':
    main()
