"""
run.py
------
Part (c): Empirical comparison of three policies over N=10,000 episodes.
Part (d): Robustness to observation noise across sigma in {0, 0.5, 1.0, 2.0}.

Outputs
-------
- Console tables with mean utility ± SE and fallback fraction.
- Figure 1: return histograms for the three policies (sigma=0).
- Figure 2: mean utility vs sigma for each policy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from env import ApartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy

T        = 4
K        = 4
N        = 10_000
SEED     = 2024
SIGMAS   = [0.0, 0.5, 1.0, 2.0]
U_MINS   = [1, 2, 3, 4]
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# Helper: run N episodes and return array of returns 
def run_episodes(policy, env_kwargs: dict, n: int, base_seed: int) -> np.ndarray:
    """Run `n` episodes; return 1-D array of per-episode total rewards."""
    rewards = np.zeros(n, dtype=np.float32)
    for i in range(n):
        env = ApartmentEnv(**env_kwargs)
        obs, info = env.reset(seed=base_seed + i)
        done = False
        ep_reward = 0.0
        while not done:
            action = policy.act(obs)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            done = terminated or truncated
        rewards[i] = ep_reward
    return rewards


def mean_se(arr):
    return arr.mean(), arr.std() / np.sqrt(len(arr))


def fallback_frac(arr):
    """Fraction of episodes where the agent got 0 (rejected everything)."""
    return (arr == 0).mean()


# 
#  PART (c) — noiseless comparison
# 
def part_c():
    print("\n" + "=" * 65)
    print("PART (c): Noiseless policy comparison  (sigma=0, N=10,000)")
    print("=" * 65)

    env_kw = {"T": T, "K": K, "noise_std": 0.0}

    # --- Threshold sweep ---
    print("\nThreshold sweep:")
    print(f"  {'u_min':>6}  {'Mean utility':>13}  {'SE':>7}  {'Fallback %':>11}")
    print("  " + "-" * 42)
    thresh_results = {}
    for u_min in U_MINS:
        pol = ThresholdPolicy(u_min=u_min)
        rew = run_episodes(pol, env_kw, N, SEED)
        m, se = mean_se(rew)
        fb    = fallback_frac(rew) * 100
        thresh_results[u_min] = rew
        print(f"  {u_min:>6}  {m:>13.4f}  {se:>7.4f}  {fb:>10.2f}%")

    best_u = max(U_MINS, key=lambda u: thresh_results[u].mean())
    print(f"\n  Best fixed threshold: u_min = {best_u}")

    # --- Three main policies ---
    policies = {
        "Random        ": RandomPolicy(T=T, seed=SEED),
        f"Threshold(u={best_u})": ThresholdPolicy(u_min=best_u),
        "Optimal       ": OptimalPolicy(),
    }

    print("\nMain policy comparison:")
    print(f"  {'Policy':20}  {'Mean utility':>13}  {'SE':>7}  {'Fallback %':>11}")
    print("  " + "-" * 57)

    all_rewards = {}
    for name, pol in policies.items():
        rew = run_episodes(pol, env_kw, N, SEED)
        m, se = mean_se(rew)
        fb    = fallback_frac(rew) * 100
        all_rewards[name.strip()] = rew
        print(f"  {name:20}  {m:>13.4f}  {se:>7.4f}  {fb:>10.2f}%")

    # --- Figure 1: return histograms ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    colours   = ["#e06c75", "#61afef", "#98c379"]
    labels    = list(all_rewards.keys())

    bins = np.arange(-0.5, K + 1.5, 1)
    for ax, (name, rew), col in zip(axes, all_rewards.items(), colours):
        ax.hist(rew, bins=bins, color=col, edgecolor="white", linewidth=0.6, alpha=0.85)
        m, se = mean_se(rew)
        ax.axvline(m, color="#2c3e50", linewidth=2, linestyle="--", label=f"Mean = {m:.3f}")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode return", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Return distributions — ApartmentEnv (T={T}, K={K}, N={N:,})",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig1_histograms.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure 1 saved → {fig_path}")

    return thresh_results, all_rewards, best_u


# 
#  PART (d) — noise robustness
# 

def part_d(best_u_min: int):
    print("\n" + "=" * 65)
    print("PART (d): Noise robustness  (sigma ∈ {0, 0.5, 1.0, 2.0})")
    print("=" * 65)

    noise_results = {
        "Random":          {s: None for s in SIGMAS},
        f"Threshold(u={best_u_min})": {s: None for s in SIGMAS},
        "Optimal":         {s: None for s in SIGMAS},
    }

    header = f"  {'sigma':>6}  " + "  ".join(f"{'Mean (' + k + ')':>22}" for k in noise_results)
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))

    for sigma in SIGMAS:
        env_kw = {"T": T, "K": K, "noise_std": sigma}
        row_vals = {}

        pol_map = {
            "Random":              RandomPolicy(T=T, seed=SEED),
            f"Threshold(u={best_u_min})": ThresholdPolicy(u_min=best_u_min),
            "Optimal":             OptimalPolicy(),
        }

        for name, pol in pol_map.items():
            rew = run_episodes(pol, env_kw, N, SEED)
            m, se = mean_se(rew)
            noise_results[name][sigma] = (m, se, rew)
            row_vals[name] = (m, se)

        row_str = f"  {sigma:>6.1f}  "
        for name in noise_results:
            m, se = row_vals[name]
            row_str += f"  {m:>10.4f} ± {se:.4f}          "
        print(row_str)

    # --- Figure 2: mean utility vs sigma ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {"Random": "#e06c75",
               f"Threshold(u={best_u_min})": "#61afef",
               "Optimal": "#98c379"}
    markers = {"Random": "o",
               f"Threshold(u={best_u_min})": "s",
               "Optimal": "D"}

    for name in noise_results:
        xs  = SIGMAS
        ys  = [noise_results[name][s][0] for s in SIGMAS]
        ses = [noise_results[name][s][1] for s in SIGMAS]
        ax.errorbar(xs, ys, yerr=ses,
                    label=name,
                    color=colours[name],
                    marker=markers[name],
                    markersize=7,
                    linewidth=2,
                    capsize=4,
                    capthick=1.5)

    ax.set_xlabel("Observation noise σ", fontsize=12)
    ax.set_ylabel("Mean episode utility", fontsize=12)
    ax.set_title(
        f"Policy robustness to observation noise\n(T={T}, K={K}, N={N:,} episodes)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(SIGMAS)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig2_noise_robustness.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure 2 saved → {fig_path}")

    return noise_results



if __name__ == "__main__":
    thresh_results, all_rewards, best_u = part_c()
    noise_results = part_d(best_u)
    print("\nAll done.")
