"""Actor and Critic networks for the SAC policy (Algorithm 1).

Action space: an N x (N+1) matrix, row i a probability distribution over
N Edge Hosts + Cloud (see /PROTOCOL.md section 3). The naive "Gaussian +
softplus + divide-by-row-sum" approach is NOT used here: that
normalization step is not invertible (many raw vectors map to the same
normalized row), so the log-density needed for SAC's entropy term cannot
be computed exactly.

Instead, each row is modeled directly as a Dirichlet distribution, whose
samples are non-negative and sum to 1 by construction -- no separate
normalization step, and PyTorch's Dirichlet gives an exact, reparameterized
log_prob for free.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

# FIX (2026-09-27, diag12/13, see Q1 report mục 20): this comment's own
# prediction ("log_prob can blow up") was realized in diag12 -- log_prob
# reached 1000+ (vs the normal ~25-51 range) once the actor got pushed
# toward a corner where several entries sat near the OLD floor of 1e-3.
# Any concentration parameter alpha_i < 1 makes the Dirichlet density's
# x_i^(alpha_i - 1) term have a NEGATIVE exponent, which diverges as the
# sampled x_i -> 0 -- a real mathematical singularity, not just numerical
# noise, and near-certain to be hit for a coordinate whose concentration is
# genuinely tiny (its samples cluster right at the boundary where the
# density blows up). Raising the floor to 1.0 keeps every alpha_i >= 1,
# which keeps that exponent >= 0 everywhere -- the density is then bounded
# (in fact vanishes, not diverges) at every simplex boundary, closing this
# off structurally rather than hoping training never wanders into the
# danger zone. This also makes CONCENTRATION_SUM_CEILING's floor-derived
# alpha_0 minimum consistent: num_cols=15 entries at floor=1.0 gives a
# minimum alpha_0 of 15 -- exactly the uniform Dirichlet(1,...,1) already
# observed empirically as where the actor settles when it has no
# differentiating signal (diag4/diag6/diag8/diag9), not a coincidence now
# that the floor is set to match.
CONCENTRATION_FLOOR = 1.0

# Upper bound on each row's CONCENTRATION SUM (alpha_0 = sum of the row's
# concentration parameters) -- NOT a per-entry cap. Differential (continuous)
# log-density is NOT bounded above the way discrete log-probability is: the
# Dirichlet's peak density grows with alpha_0 even when every entry scales up
# together (e.g. all entries equal but large still collapses the distribution
# to a spike at the simplex center as alpha_0 -> infinity), so a per-entry
# clamp alone doesn't bound achievable log_prob -- two real full training
# runs confirmed this: capping each entry at 50 (allowing alpha_0 up to
# 50*15=750) left alpha climbing on almost the exact same exponential curve
# as with no cap at all (~1.9x every 20 episodes, still ~18 by episode 105).
# Bounding alpha_0 directly bounds log Gamma(alpha_0) growth, which is what
# actually drives peak log-density -- 30 (~2x the uniform Dirichlet's
# alpha_0=15) still allows meaningfully confident rows while keeping
# log Gamma(alpha_0) small and stable (~71 vs ~4605 at alpha_0=750).
# FIX (2026-09-27, diag14, see Q1 report mục 21): diag13 (floor=1.0, alpha
# fixed/pinned at 0.0224) showed this ceiling being reached progressively
# more often over ~130+ episodes (clip_frac 0.22->0.49) even with alpha's
# SAC-temperature entropy pressure held constant -- the actor's own
# concentration output was slowly, then increasingly quickly, sharpening on
# its own (a Q_pi-driven effect, not an entropy-tuning artifact -- alpha
# itself never moved), correlating with rising drop_total and falling
# edge_balance once clip_frac climbed past ~0.35-0.4. Tightened from 30 to
# 18 (~1.2x uniform Dirichlet's alpha_0=15, down from ~2x) to bound how
# confident/peaked the policy can ever become, directly targeting the
# observed proximate mechanism instead of the entropy-tuning machinery.
CONCENTRATION_SUM_CEILING = 18.0


class Actor(nn.Module):
    def __init__(self, state_dim, num_nodes, hidden_dim=256):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_cols = num_nodes + 1  # N Edge Hosts + Cloud
        self.action_dim = num_nodes * self.num_cols

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.concentration_head = nn.Linear(hidden_dim, self.action_dim)

    def _concentration(self, state):
        raw = self.concentration_head(self.trunk(state))
        raw = raw.view(-1, self.num_nodes, self.num_cols)
        conc = F.softplus(raw) + CONCENTRATION_FLOOR
        # Rescale rows whose sum exceeds the ceiling back down to it,
        # preserving the relative shape (each entry's share of the row) --
        # only shrinks (scale <= 1), never inflates an already-modest row.
        row_sum = conc.sum(dim=-1, keepdim=True)
        scale = torch.clamp(CONCENTRATION_SUM_CEILING / row_sum, max=1.0)
        return conc * scale, row_sum

    def sample(self, state):
        """Reparameterized sample. Returns (matrix[B,N,N+1], log_prob[B],
        conc_diag, row_sum[B,N,1]).

        row_sum (2026-09-27, diag15, see agent.py's concentration-penalty
        comment): the RAW pre-ceiling alpha_0 as a gradient-carrying tensor,
        separate from conc_diag's plain-float version below (computed under
        no_grad, for logging only). The hard ceiling in _concentration()
        bounds the FORWARD value used for routing but gives the network no
        gradient telling it "stop pushing raw output higher" -- diag14
        showed alpha0 climbing unbounded (16->54) behind a permanently
        saturated clip_frac=1.0 ceiling, with no cost to the optimizer for
        continuing to push. Returning this tensor lets agent.py add an
        explicit penalty term with a real gradient back into
        concentration_head, instead of relying solely on the forward-only
        clamp.
        """
        concentration, row_sum = self._concentration(state)
        dist = Dirichlet(concentration)
        action = dist.rsample()
        # log_prob is per-row (batch, N); rows are conditionally independent
        # given the state. Use MEAN across rows, not sum: differential
        # (continuous) log-density is unbounded and each row's typical
        # magnitude is already large (tens, given CONCENTRATION_SUM_CEILING),
        # so summing N=num_nodes independent rows inflates the entropy bonus
        # -alpha*log_prob (used in both the critic's Bellman target and the
        # actor/alpha losses, see agent.py) by a factor of N -- confirmed via
        # a real 900-episode checkpoint where Q1/Q2 had diverged to ~-96,000
        # against a theoretical reward-driven steady state of ~-290 (a ~330x
        # blowup), while Q varied by <1% across wildly different candidate
        # actions on the same state -- i.e. the critic had no usable
        # action-dependent signal left, only entropy-driven noise. Mean
        # keeps the entropy bonus on a fixed, per-decision scale regardless
        # of how many hosts N the topology has.
        log_prob = dist.log_prob(action).mean(dim=-1)
        # DIAGNOSTIC (2026-09-25, see Q1 report mục 15): raw PRE-ceiling
        # alpha_0 (row_sum) and how often CONCENTRATION_SUM_CEILING=30 is
        # actually being hit -- the actor_grad collapse observed empirically
        # across all 3 completed device levels (300/600/900, actor_grad
        # ~1-3 -> ~0.15-0.2 within the first ~100 episodes and never
        # recovering, even at 900 where drop_total demonstrably responds to
        # something real) is consistent with training living mostly in the
        # clamp's zero-local-gradient region. clip_frac==1.0 essentially all
        # the time would confirm that directly; clip_frac staying near 0
        # would rule it out and point elsewhere (e.g. the critic's own
        # ∂Q/∂a going flat).
        with torch.no_grad():
            conc_diag = {
                "alpha0_mean": row_sum.mean().item(),
                "alpha0_max": row_sum.max().item(),
                "clip_frac": (row_sum.squeeze(-1) > CONCENTRATION_SUM_CEILING).float().mean().item(),
                "conc_entry_max": concentration.max().item(),
            }
        return action, log_prob, conc_diag, row_sum

    def deterministic(self, state):
        """Mean of each row's Dirichlet -- used for evaluation, not training."""
        concentration, _ = self._concentration(state)
        return concentration / concentration.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """Twin Q-networks (Q_phi1, Q_phi2)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action_flat):
        x = torch.cat([state, action_flat], dim=-1)
        return self.q1(x), self.q2(x)
