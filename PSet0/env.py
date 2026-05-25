"""
env.py
------
ApartmentEnv: a finite-horizon MDP matching 1(a).

State  : (t, U_t)  — current week and apartment quality.
Action : 0 = reject,  1 = accept
Reward : U_t on accept, 0 on reject, 0 on fallback (rejected everything).

With noise_std > 0 the *observation* of quality is U_t + N(0, sigma^2),
but the reward is always the true U_t.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ApartmentEnv(gym.Env):
    """
    Finite-horizon apartment search MDP.

    Parameters
    ----------
    T : int
        Number of weeks (horizon length).
    K : int
        Maximum quality level.
    noise_std : float
        Standard deviation of Gaussian observation noise added to quality.
        0 (default) = fully observed.
    seed : int or None
        Optional RNG seed passed to reset().
    """

    metadata = {"render_modes": []}

    def __init__(self, T: int = 4, K: int = 4, noise_std: float = 0.0, seed=None):
        super().__init__()
        self.T = T
        self.K = K
        self.noise_std = noise_std

        # Action space: 0 = reject, 1 = accept
        self.action_space = spaces.Discrete(2)

        # Observation space: (t, observed_quality)
        # t  in [1, T]  
        # observed quality in [1-6*sigma, K+6*sigma] when noisy, else [1, K]
        q_low  = 1.0 - 6.0 * noise_std
        q_high = float(K) + 6.0 * noise_std
        self.observation_space = spaces.Dict({
            "t": spaces.Discrete(T + 1),          # 0 … T  (0 = done sentinel)
            "quality": spaces.Box(
                low=np.array([q_low],  dtype=np.float32),
                high=np.array([q_high], dtype=np.float32),
                dtype=np.float32,
            ),
        })

        
        self._np_rng = np.random.default_rng(seed)

        
        self._t: int = 0
        self._true_quality: int = 0
        self._terminated: bool = False

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        self._t = 1
        self._terminated = False
        self._true_quality = int(self._np_rng.integers(1, self.K + 1))

        obs  = self._make_obs()
        info = {"true_quality": self._true_quality}
        return obs, info

    def step(self, action: int):
        assert not self._terminated, "Call reset() before stepping a terminated episode."
        assert self.action_space.contains(action), f"Invalid action {action}"

        t = self._t
        u = self._true_quality

        if action == 1:
            # Accept: receive true quality, episode ends
            reward     = float(u)
            terminated = True
            truncated  = False
            self._terminated = True
            obs  = self._make_obs(terminal=True)
            info = {"true_quality": u, "week": t, "accepted": True}

        else:
            # Reject
            reward = 0.0
            if t == self.T:
                # Last week rejected → fallback (utility 0)
                terminated = True
                truncated  = False
                self._terminated = True
                obs  = self._make_obs(terminal=True)
                info = {"true_quality": u, "week": t, "accepted": False, "fallback": True}
            else:
                # Move to next week
                terminated = False
                truncated  = False
                self._t += 1
                self._true_quality = int(self._np_rng.integers(1, self.K + 1))
                obs  = self._make_obs()
                info = {"true_quality": self._true_quality, "week": self._t, "accepted": False}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # no rendering needed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_obs(self, terminal: bool = False):
        """Build the observation dict, adding noise to quality if requested."""
        if terminal:
            return {
                "t": 0,
                "quality": np.array([0.0], dtype=np.float32),
            }
        q_obs = float(self._true_quality)
        if self.noise_std > 0.0:
            q_obs += float(self._np_rng.normal(0.0, self.noise_std))
        return {
            "t": self._t,
            "quality": np.array([q_obs], dtype=np.float32),
        }
