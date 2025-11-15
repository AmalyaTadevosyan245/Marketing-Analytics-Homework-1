import numpy as np
import pandas as pd

class EpsilonGreedy:
    """Epsilon-Greedy algorithm with decaying exploration rate."""

    def __init__(self, arms, epsilon=1.0):
        self.arms = np.array(arms)
        self.n_arms = len(arms)
        self.epsilon = epsilon
        self.counts = np.zeros(self.n_arms)
        self.estimates = np.zeros(self.n_arms)
        self.arm_history = []
        self.reward_history = []

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon:.3f})"

    def pull(self):
        if np.random.rand() < self.epsilon:
            arm = np.random.choice(self.n_arms)
        else:
            arm = np.argmax(self.estimates)
        reward = np.random.normal(self.arms[arm], 1)
        return arm, reward

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

    def experiment(self, n_trials=20000):
        for t in range(1, n_trials + 1):
            self.epsilon = 1 / t
            arm, reward = self.pull()
            self.update(arm, reward)
            self.arm_history.append(arm)
            self.reward_history.append(reward)

        regrets = np.max(self.arms) - np.array([self.arms[a] for a in self.arm_history])
        return pd.DataFrame({
            "Arm": self.arm_history,
            "Reward": self.reward_history,
            "Regret": regrets,
            "Algorithm": "Epsilon-Greedy"
        })

    def report(self):
        avg_reward = np.mean(self.reward_history)
        avg_regret = np.mean(np.max(self.arms) - np.array(self.reward_history))
        print(f"Epsilon-Greedy → Avg Reward: {avg_reward:.4f}, Avg Regret: {avg_regret:.4f}")
