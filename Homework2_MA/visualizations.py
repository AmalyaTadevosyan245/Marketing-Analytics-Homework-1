import matplotlib.pyplot as plt

class Visualization:
    """Visualization utilities for bandit experiments."""

    def __init__(self, df_eg, df_ts):
        self.df_eg = df_eg
        self.df_ts = df_ts

    def plot_learning_curve(self):
        plt.figure()
        plt.plot(self.df_eg["Reward"].rolling(200).mean(), label="Epsilon-Greedy")
        plt.plot(self.df_ts["Reward"].rolling(200).mean(), label="Thompson Sampling")
        plt.title("Learning Curve (Rolling Mean of Rewards)")
        plt.xlabel("Trial")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()

    def plot_cumulative(self):
        plt.figure()
        plt.plot(self.df_eg["Reward"].cumsum(), label="Epsilon-Greedy")
        plt.plot(self.df_ts["Reward"].cumsum(), label="Thompson Sampling")
        plt.title("Cumulative Reward")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.df_eg["Regret"].cumsum(), label="Epsilon-Greedy")
        plt.plot(self.df_ts["Regret"].cumsum(), label="Thompson Sampling")
        plt.title("Cumulative Regret")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.show()
