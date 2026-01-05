# DQN vs PPO on CartPole and LunarLander



## Overview



This repository contains a comparative reinforcement learning study between **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)**. The goal is to analyze how value based and actor critic methods differ in performance, stability, reward sensitivity, and computational efficiency across environments of increasing complexity.



The project was designed to meet the course requirements for:



* multiple RL algorithms,

* multiple environments,

* reward function analysis,

* systematic hyperparameter tuning,

* multi-seed evaluation.



---



## Environments



### CartPole-v1



* Discrete action space

* Low-dimensional state space

* Dense reward

* Used as a simple benchmark to study convergence speed and stability



### LunarLander-v3



* Discrete action space with continuous state variables

* Sparse terminal reward with shaped intermediate rewards

* More complex dynamics and sensitive to reward design



---



## Algorithms



### Deep Q-Network (DQN)



* Value based method

* Learns an action value (Q) function

* Uses experience replay and a target network

* Sensitive to hyperparameters and exploration schedule



### Proximal Policy Optimization (PPO)



* Actor critic method

* Directly optimizes a stochastic policy

* Uses clipped policy updates for stability

* Generally more robust across environments



These two algorithms represent fundamentally different RL paradigms and form the basis of the comparison.



---



## Reward Functions



Three reward variants were evaluated:



* **v0 (baseline)**: environment default reward

* **v1**: mild state-based shaping

* **v2**: refined shaping applied only in critical states



Reward shaping was treated as an **ablation study**: for fixed hyperparameters, performance under different reward variants was compared.



Key observations:



* Reward shaping had little effect on CartPole

* Reward shaping (v2) significantly improved performance and stability on LunarLander



---



## Hyperparameter Tuning



Limited but systematic **random search** was performed over key parameters:



* learning rate

* discount factor (gamma)

* exploration schedule (DQN)

* target network update interval

* neural network architecture



Each configuration was evaluated over 20–30 evaluation episodes. The best configuration was selected based on mean evaluation return.



An important finding was that single-seed optimal configurations did not always generalize across seeds, especially for DQN. All final results therefore use multi-seed evaluation.



---



## Experimental Setup



* **Framework:** Stable-Baselines3

* **Language:** Python

* **Seeds:** 5 (0–4)

* **Evaluation metric:** mean ± standard deviation of episode return

* **Training horizon:**



&nbsp; * CartPole-v1: up to 600k timesteps

&nbsp; * LunarLander-v3: 600k timesteps

* **Hardware:** local machine (CPU)



Videos were recorded using **seed 0** as a representative trained policy. Quantitative results are reported as multi-seed averages.



---



## Results Summary



### CartPole-v1



| Algorithm | Reward | Mean ± Std    |

| --------- | ------ | ------------- |

| PPO       | v0     | 500.0 ± 0.0   |

| DQN       | v0     | 343.7 ± 191.4 |



PPO solved CartPole consistently across all seeds. DQN exhibited high variance despite tuning and extended training.



### LunarLander-v3



| Algorithm | Reward | Mean ± Std   | Train Time |

| --------- | ------ | ------------ | ---------- |

| PPO       | v2     | 264.2 ± 9.8  | ~760 s     |

| DQN       | v2     | 245.0 ± 20.6 | ~1450 s    |



On LunarLander, PPO achieved higher return with lower variance and approximately half the training time.



---



## Key Insights



* Actor critic methods (PPO) are more stable and sample efficient than value based methods (DQN)

* Reward shaping is environment dependent and most effective in complex tasks

* Multi seed evaluation is essential for reliable RL comparison

* Hyperparameter tuning alone is insufficient to eliminate instability in vanilla DQN





---



## Reproducibility



All experiments are reproducible using the provided configuration files and scripts. Multi seed evaluation and aggregated results ensure robustness of conclusions.



---

## How to Run

### 1. Create virtual environment

python -m venv .venv
# Windows
.\.venv\Scripts\activate

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run experiments
python -m src.run_all --configs_dir configs --runs_dir runs_fresh --seeds 0 1 2 3 4

### 4. Aggregate results
python -m src.aggregate_results --runs_dir runs_fresh --out results/aggregate.json


---



## Conclusion



This project demonstrates that algorithm choice, reward design, and evaluation methodology are critical in reinforcement learning. PPO consistently outperformed DQN in stability and efficiency, while reward shaping proved beneficial only in more complex environments.



The results highlight practical trade offs between value based and actor critic approaches and reinforce the importance of disciplined experimental design in RL research.



