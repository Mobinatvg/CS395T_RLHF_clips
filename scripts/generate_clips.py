import gymnasium as gym
import numpy as np
import os
import uuid

# Folder locations relative to repo root
CLIP_DIR = "../clips"
TRAJ_DIR = "../traj_data"

os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)


def collect_segments(
    env_name="Walker2d-v5",
    steps=3000,
    segment_length=20
):
    """
    Collect trajectory segments from random rollouts.
    Each segment = 20 timesteps (~0.5 sec)
    """

    env = gym.make(env_name)
    obs, _ = env.reset()

    buffer_obs = []
    buffer_act = []

    all_segments = []

    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        buffer_obs.append(obs)
        buffer_act.append(action)

        obs = next_obs

        if len(buffer_obs) >= segment_length:
            seg_id = str(uuid.uuid4())

            seg_obs = np.array(buffer_obs[-segment_length:])
            seg_act = np.array(buffer_act[-segment_length:])

            save_path = os.path.join(TRAJ_DIR, f"{seg_id}.npz")
            np.savez(save_path, obs=seg_obs, act=seg_act)

            all_segments.append(save_path)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    return all_segments


if __name__ == "__main__":
    print("Collecting segments...")
    segs = collect_segments()
    print(f"Saved {len(segs)} segments.")

