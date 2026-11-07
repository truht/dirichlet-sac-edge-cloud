"""Multi-objective reward R_t (Algorithm 1, steps 21-27).

R_t = -[ w1 * CVaR_beta(D_t) + w2 * (1 / F(E_t+1)) + w3 * P_drop^t ]

REDESIGN (2026-09-26, see Q1 report mục 16): real per-episode statistics
recomputed from the 18 finished SAC-B runs (seed1, full 400 episodes each)
showed the raw combined reward is dominated almost entirely by CVaR
regardless of device count, while drop_count -- the ONE component that
actually varies substantially with the policy -- is squeezed under 2% of
the total magnitude:

    level  cvar share   fairness share   drop share   drop rel.SD
    300    99.30%        0.70%           ~0.0001%      (no signal at all)
    600    99.17%        0.77%           0.061%         61.2%
    900    97.53%        0.76%           1.715%         30.6%

Matching w3 to a fixed target share would need w3~=38 at 900 but w3~=1095
at 600 (>25x apart) -- no single fixed weight works across device levels.
EMANorm below normalizes each raw component to ~zero-mean/unit-variance
online instead, so w1/w2/w3 become post-normalization emphasis knobs
(1:1:1 is the natural "equal say" default) rather than device-level-
specific magnitude hacks.
"""

import math

import numpy as np


def compute_cvar(latencies_ms, beta):
    """Empirical CVaR (Rockafellar-Uryasev) of the tail (1-beta) latencies.

    d_(1) <= ... <= d_(M), K = ceil(M*beta), CVaR = mean(d_(K..M)).
    M=0 (no task completed this slot) has no tail to measure -> 0.0.
    """
    if not latencies_ms:
        return 0.0
    d = sorted(latencies_ms)
    m = len(d)
    k = max(1, min(math.ceil(m * beta), m))
    tail = d[k - 1:]  # d_(K)..d_(M), 1-indexed in the paper -> k-1 in 0-indexed Python
    return sum(tail) / len(tail)


def jain_fairness(energy_values):
    """Jain's Fairness Index over a per-host vector (0(1), all-equal, N/A).

    F = (sum x_i)^2 / (N * sum x_i^2), in (0, 1], 1 = perfectly fair.
    All-zero is a genuine 0/0 -- defined as 1.0 here: with nothing to
    balance, there is nothing left to be unfair about. Historically called
    with per-host REMAINING battery (%); as of the reward redesign it is
    called with per-host CUMULATIVE DRAINED energy since episode start
    instead (see train.py) -- same function, different input vector.
    """
    e = np.asarray(energy_values, dtype=np.float64)
    total = e.sum()
    if total <= 1e-9:
        return 1.0
    return float((total ** 2) / (len(e) * np.square(e).sum()))


class EMANorm:
    """Online mean/variance (exponential moving average) for normalizing one
    raw reward component to ~zero-mean/unit-variance, so w1/w2/w3 control
    relative EMPHASIS instead of having to also compensate for a raw-scale
    mismatch that differs by >25x between device levels (see module
    docstring).

    FIX (2026-09-27, diag9 -- see Q1 report mục 18): letting this adapt for
    the ENTIRE run turned out to be actively harmful for a component the
    policy itself drives (drop_count): as the actor drifted toward a worse
    (higher-drop) policy, this tracker's own mean drifted UP to match,
    making the "new normal" look statistically unremarkable again --
    diluting exactly the penalty that should have corrected the drift, a
    self-reinforcing feedback loop. drop_total climbed from ~250-450 to
    1000-1600+ over 190 episodes under log_alpha_max=-3.8 with adaptive
    norms; the same alpha fixed the ANALOGOUS sanity-check reward (diag8)
    precisely because that reward's own driving signal, G_i, is an
    environment feature the policy barely influences back, so no such loop
    existed there. `freeze()` -- called once, right when warmup ends and
    real (non-random) actions start (see train.py) -- locks mean/var at
    whatever a genuinely RANDOM policy produced, so the reference point
    can never be dragged along by the policy it is supposed to be judging,
    in either direction (this also stops it moving in a GOOD direction if
    behavior improves later, which is an acceptable tradeoff for removing
    the runaway risk -- re-derive log_alpha_max, not this freeze point, if
    that turns out to matter).
    """

    def __init__(self, decay=0.999, eps=1e-4):
        self.decay = decay
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False
        self.frozen = False

    def update(self, x):
        if self.frozen:
            return
        if not self.initialized:
            self.mean = x
            self.initialized = True
        else:
            delta = x - self.mean
            self.mean += (1.0 - self.decay) * delta
            self.var = self.decay * self.var + (1.0 - self.decay) * delta * delta

    def freeze(self):
        self.frozen = True

    def normalize(self, x):
        return (x - self.mean) / (math.sqrt(self.var) + self.eps)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "initialized": self.initialized,
                "frozen": self.frozen}

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.var = d["var"]
        self.initialized = d["initialized"]
        self.frozen = d.get("frozen", False)  # backward-compat: pre-freeze checkpoints


class RewardNorms:
    """Bundle of the 3 EMANorm trackers, one per reward component, saved/
    loaded as a unit alongside the SAC checkpoint (see agent.py) so resuming
    after a crash/reboot doesn't reset the running statistics to zero."""

    def __init__(self, decay=0.999):
        self.cvar = EMANorm(decay)
        self.fair = EMANorm(decay)
        self.drop = EMANorm(decay)

    def freeze_all(self):
        self.cvar.freeze()
        self.fair.freeze()
        self.drop.freeze()

    def state_dict(self):
        return {
            "cvar": self.cvar.state_dict(),
            "fair": self.fair.state_dict(),
            "drop": self.drop.state_dict(),
        }

    def load_state_dict(self, d):
        self.cvar.load_state_dict(d["cvar"])
        self.fair.load_state_dict(d["fair"])
        self.drop.load_state_dict(d["drop"])


def compute_reward(latencies_ms, energy_values, drop_count, beta, w1, w2, w3, norms):
    """Returns (reward, cvar, fairness, norm_cvar, norm_inv_fair, norm_drop).

    cvar/fairness stay in their original physical units for logging and
    for the caller's own diagnostics (unaffected by normalization); reward
    itself is now built from the NORMALIZED components. norms is a
    RewardNorms instance -- one per training run, persisted across
    episodes/resumes (see train.py + agent.py).
    """
    cvar = compute_cvar(latencies_ms, beta)
    fairness = jain_fairness(energy_values)
    inv_fair = 1.0 / fairness

    norms.cvar.update(cvar)
    norms.fair.update(inv_fair)
    norms.drop.update(drop_count)

    norm_cvar = norms.cvar.normalize(cvar)
    norm_inv_fair = norms.fair.normalize(inv_fair)
    norm_drop = norms.drop.normalize(drop_count)

    reward = -(w1 * norm_cvar + w2 * norm_inv_fair + w3 * norm_drop)
    return reward, cvar, fairness, norm_cvar, norm_inv_fair, norm_drop
