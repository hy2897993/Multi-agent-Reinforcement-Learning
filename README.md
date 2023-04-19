# Multi-agent Reinforcement Learning in Football Environment

Reinforcement learning with a multiagent system is a more complex problem than single-agent reinforcement learning. In multi-agent RL we need to achieve not only maximize the reward but studying how multiple agents interact in the same environment. The interaction between agents can be cooperation, competition, or mixed, depending on the environment and training goal. In this report, I experimented with several multi-agent RL algorithms, including Proximal Policy Optimization (PPO) and Importance-weighted Actor-Learner Architecture (IMPALA), in the google research Football environment. My focus is on maximizing the reward while improving the cooperation behavior of players. To compare these algorithms, I analyzed the training results of the trained agents, and the learning metrics and agents’ behavior statics during training.


- **Implement Proximal Policy Optimization (PPO) algorithm which utilize a novel "surrogate" objective function using stochastic gradient ascent.**
- **Use an actor-critic method with a centralized critic to learn and decentralized actors to optimize the agents’ policies.**
- **Implement Deep-RL with Importance Weighted Actor-Learner Architectures (IMPALA) which provides high throughput in singlemachine training and is able to scale to multiple machines without data efficiency loss.**
- **Apply off-policy correction method V-trace and decoupled acting to achieve stable learning.**


