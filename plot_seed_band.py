import glob
import numpy as np
import matplotlib.pyplot as plt

def load_curves(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    timesteps = None
    curves = []
    for p in paths:
        d = np.load(p)
        t = d["timesteps"]
        r = d["results"].mean(axis=1)
        if timesteps is None:
            timesteps = t
        else:
            # ensure same eval schedule
            if len(t) != len(timesteps) or not np.all(t == timesteps):
                raise ValueError(f"Timesteps mismatch for {p}. Plot them separately or resample.")
        curves.append(r)
    return timesteps, np.vstack(curves), paths

def plot_band(ax, t, R, label):
    med = np.median(R, axis=0)
    p25 = np.percentile(R, 25, axis=0)
    p75 = np.percentile(R, 75, axis=0)
    ax.plot(t, med, label=label, linewidth=2)
    ax.fill_between(t, p25, p75, alpha=0.2)

t, R, _ = load_curves(r"results\cartpole-v1_ppo_v0_seed*\eval\evaluations.npz")

fig = plt.figure(figsize=(7,4))
ax = plt.gca()
plot_band(ax, t, R, "PPO (v0) median ± IQR")

ax.set_xlabel("Timesteps")
ax.set_ylabel("Mean Episode Return")
ax.set_title("CARTPOLE-v1: PPO multi-seed learning curve")
ax.set_ylim(0, 520)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig("CARTPOLE-v1_PP0_multiseed.png", dpi=200)
plt.show()
