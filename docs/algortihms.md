algorithms.py yields 4 different implemetaton of **Tempporal Difference** algorithms, namely,  
Q-Learn
Q-Learn with Constant Depth Search
Q-Learn with Varying Depth Search
Sarsa

This file have 5 global variables to tune experiments. These are: 
- max_episode: Maximum number of tries the agent will go through
- learning_rate: Speed of acquiring newly calculated values
- discount: discount rate in Bellman equation. Between 0-1. High discount rate means agent favors immediate reward over time value.
- epsilon: Epsilon is the trade-off between exploration.
- decay_rate: The change in epsilon. 