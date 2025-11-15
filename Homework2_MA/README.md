## Overview


This project simulates a multi-armed bandit problem with four arms having reward means [1, 2, 3, 4].  
The goal is to compare two popular reinforcement learning algorithms:


1. Epsilon-Greedy — explores with probability ε, which decays over time (ε = 1/t)  
2. Thompson Sampling — samples from a Bayesian posterior (Normal-Normal model)


Each algorithm runs for 20,000 trials. The code tracks rewards, computes regret, and visualizes the learning process.


---


## How to Run


1. Open the `experiment.ipynb` notebook in Jupyter Notebook.
2. Run all cells in sequence to:
   - Initialize both algorithms  
   - Execute 20,000 simulated trials  
   - Plot learning curves, cumulative rewards, and cumulative regrets  
   - Save results to `results.csv`
## Output


After execution, the following results are produced:


- Console output:
  - Average reward and average regret for each algorithm


- Plots:
  - Learning curve (rolling average of rewards)
  - Cumulative reward comparison
  - Cumulative regret comparison


- CSV file (`results.csv`):
  - Contains the following columns:
    - Arm  
    - Reward  
    - Regret  
    - Algorithm


---


## Interpretation


| Algorithm | Behavior | Description |
|------------|-----------|--------------|
| Epsilon-Greedy | Starts fully random, gradually focuses on best arm | Simple and intuitive but may over-explore early |
| Thompson Sampling | Uses Bayesian updates for exploration | More adaptive and data-efficient |


---


## Suggested Improvements (Bonus)


- Test with multiple random seeds and compare average performance  
- Implement UCB (Upper Confidence Bound) as a third baseline  
- Run experiments with different reward distributions  
- Visualize confidence intervals over multiple runs
