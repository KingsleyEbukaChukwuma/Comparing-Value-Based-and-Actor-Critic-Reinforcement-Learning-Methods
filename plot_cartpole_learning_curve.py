import numpy as np
import matplotlib.pyplot as plt

ppo = np.load(
    r"results\cartpole-v1_ppo_v0_seed0_1767374681\eval\evaluations.npz"
)
dqn = np.load(
    r"runs_cartpole_dqn_robust\cartpole-v1_dqn_v0_seed0_1767520668\eval\evaluations.npz"
)

ppo_t = ppo["timesteps"]
ppo_r = ppo["results"].mean(axis=1)

dqn_t = dqn["timesteps"]
dqn_r = dqn["results"].mean(axis=1)

# truncate DQN to PPO horizon
max_t = ppo_t.max()
mask = dqn_t <= max_t

plt.figure(figsize=(7,4))
plt.plot(ppo_t, ppo_r, label="PPO", linewidth=2)
plt.plot(dqn_t[mask], dqn_r[mask], label="DQN", linewidth=2)

plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Return")
plt.title("CartPole-v1 Learning Curves (Early Training)")
plt.ylim(0, 520)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cartpole_learning_curve.png", dpi=200)
plt.show()
