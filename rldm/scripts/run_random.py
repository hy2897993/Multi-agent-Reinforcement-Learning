import argparse

from rldm.utils import football_tools


def rollout(num_episodes:int):
    env = football_tools.RllibGFootball('3_vs_3_auto_GK')

    for i in range(num_episodes):
        observations = env.reset()

        done, ep_reward = False, {a: 0 for a in observations}
        while not done:
            random_actions = {a: env.action_space[a].sample() for a in env.action_space}
            observations, rewards, dones, infos = env.step(random_actions)
            ep_reward = {a: ep_reward[a] + rewards[a] for a in rewards}
            done = dones['__all__']
        print(f"Episode {i} completed.")
        print("Rewards:")
        for a, ep_r in ep_reward.items():
            print(f"\t{a}: {ep_r}")
        print(f"\ttotal: {sum(ep_reward.values())}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rollout a random policy on the football environment')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to roll out')
    args = parser.parse_args()
    rollout(args.num_episodes)
