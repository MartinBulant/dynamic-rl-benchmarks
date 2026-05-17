import numpy as np
import argparse
import imageio
import gymnasium
import dyn_rl_benchmarks  # noqa: F401, do not remove, it is for gym registration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo environment with random actions."
    )
    parser.add_argument(
        "--env", default="Drawbridge-v1", help="Name of gym environment to be run."
    )
    args = parser.parse_args()

    env = gymnasium.make(args.env)

    obs, info = env.reset()
    done = False
    frames = []
    while not done:
        action = env.action_space.sample()
        obs, rew, terminate, truncated, info = env.step(action)
        done = terminate or truncated
        img = env.render()
        frames.append(img)

    imageio.mimsave("video.mp4", frames, fps=30)
