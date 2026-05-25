"""
policies.py
-----------
Three policies for ApartmentEnv.

Each policy exposes:
    act(obs) -> int   (0 = reject, 1 = accept)

where obs is the dict returned by ApartmentEnv.reset / .step:
    {"t": int, "quality": np.ndarray([q_observed])}
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
#  1. Random Policy
# ──────────────────────────────────────────────────────────────

class RandomPolicy:
    """
    Accept with probability 1/T each week; otherwise reject.

    """

    def __init__(self, T: int, seed=None):
        self.T = T
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> int:
        return int(self._rng.random() < 1.0 / self.T)

    def __repr__(self):
        return f"RandomPolicy(T={self.T})"


# ──────────────────────────────────────────────────────────────
#  2. Threshold Policy
# ──────────────────────────────────────────────────────────────

class ThresholdPolicy:
    """
    Accept iff the observed quality >= u_min.
    
    """

    def __init__(self, u_min: float):
        self.u_min = u_min

    def act(self, obs: dict) -> int:
        q_observed = float(obs["quality"][0])
        return int(q_observed >= self.u_min)

    def __repr__(self):
        return f"ThresholdPolicy(u_min={self.u_min})"


# ──────────────────────────────────────────────────────────────
#  3. Optimal Policy (from Problem 1(c), T=4, K=4)
# ──────────────────────────────────────────────────────────────

class OptimalPolicy:
    """
    Hard-coded optimal policy for T=4, K=4, computed by backwards.

    Accept iff true_quality >= threshold[t]:
        t=1: threshold = 4    (accept only u=4)
        t=2: threshold = 3    (accept u in {3,4})
        t=3: threshold = 3    (accept u in {3,4})
        t=4: threshold = 1    (always accept — last chance)

    In the noisy setting the agent compares the *observed* quality to
    the threshold, since the true quality is hidden.  This means the
    policy can make errors when noise is large.

    Lookup table: _TABLE[(t, u_true)] -> action  (noiseless case).
    In the noisy path we threshold on the observed quality.
    """

    # Thresholds derived from W_{t+1} values:
    
    _THRESHOLDS = {1: 4, 2: 3, 3: 3, 4: 1}

    def act(self, obs: dict) -> int:
        t          = int(obs["t"])
        q_observed = float(obs["quality"][0])

        if t == 0:
            return 0

        threshold = self._THRESHOLDS.get(t, 1)
        return int(q_observed >= threshold)

    def __repr__(self):
        return "OptimalPolicy(T=4, K=4)"
