import argparse

import gymnasium

import dyn_rl_benchmarks  # noqa: F401, do not remove, it is for gym registration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo environment with random actions."
    )
    parser.add_argument(
        "--env", default="Platforms-v1", help="Name of gym environment to be run."
    )
    args = parser.parse_args()

    env = gymnasium.make(args.env)

    for n in range(10):
        obs, info = env.reset()
        cum_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, rew, terminate, truncated, info = env.step(action)
            done = terminate or truncated
            env.render()
            cum_reward += rew
            if done:
                print("Episode return: {}".format(cum_reward))
