# Flocking with Selective Evolutionary Multi-Agent Reinforcement Learning (EMARL) Algorithm 



> This is the **Selective Evolutionary Multi-Agent Reinforcement Learning （SEMARL） algorithm**  implementation on Pytorch version, the corresponding paper is [Cooperation and Competition: Flocking with Evolutionary Multi-Agent Reinforcement Learning](https://www.researchgate.net/publication/369976980_Cooperation_and_Competition_Flocking_with_Evolutionary_Multi-Agent_Reinforcement_Learning) (accepted by 29th ICONIP, conference version) & [Semarl: Selective Evolutionary Multi-Agent Reinforcement Learning for Improving Cooperative Flocking with Competition](https://www.researchgate.net/publication/369344044_Semarl_Selective_Evolutionary_Multi-Agent_Reinforcement_Learning_for_Improving_Cooperative_Flocking_with_Competition) (submitted, journal version)

## Algorithms

- SEMARL (Selective Evolutionary Multi-Agent Reinforcement Learning, proposed)
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- COMA (Counterfactual Multi-Agent Policy Gradient)
- IQL (Independent Q-Learning (with DNNs))
- SQDDPG (Shapely Q-value Deep Deterministic Policy Gradient, not realized)


## Requirements

- python=3.8.5
- torch>=1.13.1


**Or download the python environment directly: [LG-CS.zip](https://pan.baidu.com/s/1ODtPNWxLOWAHcw7ZDz2sWw)
Extract code: MARL**

## Training Agents with SEMARL

If the python environment **LG-CS** is loaded, using follow instruction to train 15 agents (5 are senior agents, 5 are junior agents):

```shell
python main.py --n=15 --n-senior=5 --n-junior=5
```