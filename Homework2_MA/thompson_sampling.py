import numpy as np
import pandas as pd

class ThompsonSampling:
    """Thompson Sampling using Gaussian-Normal conjugate updates."""

    def __init__(self, arms, prior_mean=0, prior_var=1, reward_var=1):
        self.arms = np.array(arms)
        self.n_arms = len(arms)
        self.posterior_mean = np.full(self.n_arms, prior_mean, dtype=float)
        self.posterior_var = np.full(self.n_arms, prior_var, dtype=float)
        self.reward_var = reward_var
        self.arm_history = []
        self.reward_history = []

    def __repr__(self):
        return "ThompsonSampling()"

    def pull(self):
        sampled_means = np.random.normal(self.posterior_mean, np.sqrt(self.posterior_var))
        arm = np.argmax(sampled_means)
        reward = np.random.normal(self.arms[arm], np.sqrt(self.reward_var))
        return arm, reward

    def update(self, arm, reward):
        prior_m = self.posterior_mean[arm]
        prior_v = self.posterior_var[arm]

        post_v = 1 / (1/prior_v + 1/self.reward_var)
        post_m = post_v * (prior_m/prior_v + reward/self.reward_var)

        self.posterior_mean[arm] = post_m
        self.posterior_var[arm] = post_v

    def experiment(self, n_trials=20000):
        for _ in range(n_trials):
            arm, reward = self.pull()
            self.update(arm, reward)
            self.arm_history.append(arm)
            self.reward_history.append(reward)

        regrets = np.max(self.arms) - np.array([self.arms[a] for a in self.arm_history])
        return pd.DataFrame({
            "Arm": self.arm_history,
            "Reward": self.reward_history,
            "Regret": regrets,
            "Algorithm": "Thompson Sampling"
        })

    def report(self):
        avg_reward = np.mean(self.reward_history)
        avg_regret = np.mean(np.max(self.arms) - np.array(self.reward_history))
        print(f"Thompson Sampling → Avg Reward: {avg_reward:.4f}, Avg Regret: {avg_regret:.4f}")
