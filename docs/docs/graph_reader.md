Graph reader is one most the most important to the frameworks of the project

### read_from_file(filename) 
Reads given file from the disk and return its project representation

### convert_to_networkX(graph)
Converts read graph to networkX format, therefore networkX properties could be used. (For example shortest path). This function also used densely by the other source files of the projects to read graphs.

### assure_DFA(graph):
Checks properties of being a DFA, if a DFA is not defected, lets them to chill. But if matrix is not 
a DFA this could drastically effect expreriment outcomes.