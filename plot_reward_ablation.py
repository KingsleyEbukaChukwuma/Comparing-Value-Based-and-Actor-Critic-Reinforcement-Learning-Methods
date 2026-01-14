import numpy as np
import matplotlib.pyplot as plt

# ===== CARTPOLE =====
cartpole_dqn = {
    "v0": (122.65, 36.0136),
    "v1": (110.4443, 7.8507),
    "v2": (262.7083, 178.0690),
}

cartpole_ppo = {
    "v0": (500.0, 0.0),
    "v1": (499.9805, 0.00317),
    "v2": (499.9167, 0.03017),
}

# ===== LUNARLANDER =====
lunar_dqn = {
    "v0": (225.6414, 24.1775),
    "v1": (232.9565, 20.6676),
    "v2": (245.0337, 20.6222),
}

lunar_ppo = {
    "v0": (238.5775, 31.1888),
    "v1": (223.1202, 25.3684),
    "v2": (264.1676, 9.8460),
}


def barplot(data, title, ylim=None, filename="plot.png"):
    labels = list(data.keys())
    means = [data[k][0] for k in labels]
    stds  = [data[k][1] for k in labels]
    x = np.arange(len(labels))

    plt.figure(figsize=(6,4))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels)
    plt.ylabel("Mean Return (± std across seeds)")
    plt.title(title)
    if ylim: plt.ylim(*ylim)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

barplot(cartpole_dqn, "CartPole-v1: DQN reward ablation (default hyperparameters)", ylim=(0, 520), filename="cartpole_dqn_reward_ablation.png")
barplot(cartpole_ppo, "CartPole-v1: PPO reward ablation", ylim=(0, 520), filename="cartpole_ppo_reward_ablation.png")
barplot(lunar_dqn, "LunarLander-v3: DQN reward ablation", ylim=(-300, 350), filename="lunar_dqn_reward_ablation.png")
barplot(lunar_ppo, "LunarLander-v3: PPO reward ablation", ylim=(-300, 350), filename="lunar_ppo_reward_ablation.png")
