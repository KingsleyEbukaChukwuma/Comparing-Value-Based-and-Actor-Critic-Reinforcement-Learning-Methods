import numpy as np
import matplotlib.pyplot as plt

ppo = np.load(
    r"results\lunarlander-v3_ppo_v2_seed0_1767407962\eval\evaluations.npz"
)
dqn = np.load(
    r"results\lunarlander-v3_dqn_v2_seed0_1767392732\eval\evaluations.npz"
)

ppo_timesteps = ppo["timesteps"]
ppo_mean_reward = ppo["results"].mean(axis=1)

dqn_timesteps = dqn["timesteps"]
dqn_mean_reward = dqn["results"].mean(axis=1)

plt.figure(figsize=(7, 4))
plt.plot(ppo_timesteps, ppo_mean_reward, label="PPO (v2)", linewidth=2)
plt.plot(dqn_timesteps, dqn_mean_reward, label="DQN (v2)", linewidth=2)

plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Return")
plt.title("LunarLander-v3 Learning Curves: PPO vs DQN")
plt.ylim(-300, 350)  # standard LunarLander range
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("lunar_learning_curve.png", dpi=200)
plt.show()
