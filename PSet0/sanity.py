"""
sanity.py
---------
Runs one episode of ApartmentEnv with a random policy and prints
(t, U_t, action, reward, done) at each step.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from env import ApartmentEnv
from policies import RandomPolicy

def main():
    env    = ApartmentEnv(T=4, K=4, seed=42)
    policy = RandomPolicy(T=4, seed=0)

    obs, info = env.reset(seed=42)
    print("=" * 60)
    print("Sanity check — one episode with RandomPolicy")
    print(f"{'Step':>4}  {'t':>3}  {'U_t (true)':>10}  {'q_obs':>8}  "
          f"{'action':>8}  {'reward':>7}  {'done':>5}")
    print("-" * 60)

    done = False
    step = 0
    total_reward = 0.0

    while not done:
        step += 1
        t_now      = obs["t"]
        q_obs      = float(obs["quality"][0])
        true_q     = info.get("true_quality", "?")
        action     = policy.act(obs)
        action_str = "accept" if action == 1 else "reject"

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        print(f"{step:>4}  {t_now:>3}  {str(true_q):>10}  {q_obs:>8.3f}  "
              f"{action_str:>8}  {reward:>7.2f}  {str(done):>5}")

    print("-" * 60)
    print(f"Episode finished after {step} step(s).  Total reward = {total_reward:.2f}")
    print()

    #  Second episode to show different path 
    print("=" * 60)
    print("Second episode (different seed)")
    print(f"{'Step':>4}  {'t':>3}  {'U_t (true)':>10}  {'q_obs':>8}  "
          f"{'action':>8}  {'reward':>7}  {'done':>5}")
    print("-" * 60)

    env2    = ApartmentEnv(T=4, K=4, seed=7)
    policy2 = RandomPolicy(T=4, seed=99)
    obs, info = env2.reset(seed=7)

    done = False
    step = 0
    total_reward = 0.0
    while not done:
        step += 1
        t_now      = obs["t"]
        q_obs      = float(obs["quality"][0])
        true_q     = info.get("true_quality", "?")
        action     = policy2.act(obs)
        action_str = "accept" if action == 1 else "reject"

        obs, reward, terminated, truncated, info = env2.step(action)
        done = terminated or truncated
        total_reward += reward

        print(f"{step:>4}  {t_now:>3}  {str(true_q):>10}  {q_obs:>8.3f}  "
              f"{action_str:>8}  {reward:>7.2f}  {str(done):>5}")

    print("-" * 60)
    print(f"Episode finished after {step} step(s).  Total reward = {total_reward:.2f}")
    print()
    print("Sanity checks passed.")

if __name__ == "__main__":
    main()
