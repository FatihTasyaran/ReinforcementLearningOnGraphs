algorithms.py contains all algorithms implemented in the project, which are:
* Sarsa
* Q Learner
* Q Learner with constant depth
* Q Learner with varying depth

Filename (graph) to be processed is hardcoded in the source file and can be changed withni `main()`.

The corresponding functions for implementations and their parameters are as following:

### sarsa(rewards, framework_graph, graph, q_table, start)
Runs Sarsa implementation, with parameters:
* **rewards**: n = #of states 1 dimensional reward table for the environment

* **framework_graph**: generated graph, actually used here for only the faulty state information

* **graph**: graph, converted to list of lists, for ease of programming

* **q_table**: values for (s,a) pairs

* **start**: starting state for the search

Returns: 

* **total_reward**: array of sum of rewards in the q table for each episode

* **total_time_steps**: array of # of time steps to find faulty node for each episode


### q_learn(rewards, framework_graph, graph, q_table, start)
Runs q learn implementation

Returns: 

* **total_reward**: array of sum of rewards in the q table for each episode

* **total_time_steps**: array of # of time steps to find faulty node for each episode

### q_learn_constant_depth(rewards, framework_graph, graph, q_table, start)

Runs q learn with further look-up q-values as **depth**

Returns: 

* **total_reward**: array of sum of rewards in the q table for each episode

* **total_time_steps**: array of # of time steps to find faulty node for each episode

### q_learn_varying_depth(rewards, framework_graph, graph, q_table, start)

Runs q learn with further look-up q-values as **depth**. In this variant `depth` starts high and decreases over time.

Returns: 

* **total_reward**: array of sum of rewards in the q table for each episode

* **total_time_steps**: array of # of time steps to find faulty node for each episode

