"""SAC agent: Actor + twin Critics + target networks (Algorithm 1, steps
1-2 & 32-34).

REVERTED (2026-09-27, diag14, see Q1 report mục 21): automatic entropy
temperature (Lagrangian dual auto-alpha) removed entirely. Not a rushed
simplification -- diag13 showed the SAC temperature `alpha` itself was
NOT the thing drifting (it sat perfectly pinned at its clamp, e.g.
0.0224, unchanged for 150+ consecutive episodes once the clamp was hit at
ep~11); what drifted was the ACTOR'S OWN Dirichlet concentration output
(a distinct quantity, unfortunately also conventionally called "alpha" in
Dirichlet notation -- see networks.py's alpha0/CONCENTRATION_SUM_CEILING,
NOT this file's alpha/temperature). Given alpha was already effectively a
constant for the entire diag13 run in practice, the auto-tuning machinery
(log_alpha, its own optimizer, target_entropy) was contributing nothing
except surface area for subtle bugs -- removed outright rather than kept
as dead weight. Use a genuinely fixed alpha (config.alpha_fixed) instead.
The real fix for the observed drift lives in networks.py
(CONCENTRATION_SUM_CEILING tightened) and is a SEPARATE, more direct lever
on the mechanism that was actually misbehaving.
"""

import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from sac.networks import Actor, Critic


class SACAgent:
    def __init__(self, state_dim, num_nodes, hidden_dim=256,
                 gamma=0.99, tau=0.005, lr=3e-4, device="cpu",
                 critic_lr=None, grad_clip_norm=None,
                 alpha_fixed=0.0224, conc_target=17.0, conc_lambda=0.01,
                 conc_lambda_min=0.001, conc_lambda_max=1.0, conc_lambda_lr=0.001,
                 policy_delay=2, target_smoothing_sigma=0.02, target_smoothing_clip=0.05):
        # FIX (2026-09-27, diag15, see networks.py's row_sum comment + Q1
        # report mục 22): CONCENTRATION_SUM_CEILING alone (diag14) bounds
        # the FORWARD value used for routing but gives the network no
        # gradient discouraging it from continuing to push its raw output
        # ever higher behind that wall -- diag14's alpha0 climbed 16->54
        # unbounded while clip_frac sat pinned at 1.000 from ep~120 onward,
        # with drop_total eventually degrading anyway (~ep100+). This adds
        # an explicit penalty term with a REAL gradient back into
        # concentration_head, so the optimizer genuinely experiences
        # increasing resistance the further raw alpha_0 overshoots
        # conc_target -- a restoring force, not just a forward-pass clamp.
        # conc_target=17 sits just under the ceiling (18), in the range
        # empirically observed as healthy (ep11-90 of diag13/14, alpha0
        # 16-20, no drop degradation). conc_lambda is a first guess, NOT
        # yet measured -- re-derive from a real checkpoint's gradient
        # magnitudes (compare d(actor_loss)/d(concentration_head) from the
        # -q_pi term vs from this penalty) the same way every other
        # coefficient in this file has been calibrated, if this value turns
        # out too weak or too strong.
        self.device = torch.device(device)
        self.num_nodes = num_nodes
        self.num_cols = num_nodes + 1
        self.action_dim = num_nodes * self.num_cols
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.critic_lr = critic_lr if critic_lr is not None else lr
        self.grad_clip_norm = grad_clip_norm
        # Fixed SAC temperature -- no dual-gradient auto-tuning (see module
        # docstring). 0.0224 default matches diag13's own clamp value
        # (log_alpha_max=-3.8), the last empirically-observed-stable point
        # before this session moved on to investigating concentration
        # growth as the real driver instead.
        self.alpha_value = alpha_fixed
        self.conc_target = conc_target
        # ADAPTIVE (2026-09-28, diag16, see config.py's conc_lambda comment):
        # conc_lambda is now MUTABLE state, not a fixed hyperparameter --
        # updated every update() call via dual ascent on the real, observed
        # equilibrium constraint row_sum ~= conc_target (see update()'s
        # bottom block). *_min/*_max are a hard safety-net clamp; never let
        # an auto-tuned quantity drift fully unbounded again (diag9-13's
        # log_alpha lesson).
        self.conc_lambda = conc_lambda
        self.conc_lambda_min = conc_lambda_min
        self.conc_lambda_max = conc_lambda_max
        self.conc_lambda_lr = conc_lambda_lr

        # BREAKTHROUGH (2026-09-28, diag17): diag15/16 investigated the
        # concentration side of the actor as if it were the primary
        # mechanism, but a direct check on diag15's own log data revealed
        # q_pi (~-33 at ep180) running ~20-30x LARGER than what an unbiased
        # critic could produce given the actually-observed episode returns
        # (~-1 to -3) -- not legitimate value convergence, but a classic
        # off-policy actor-critic pathology: extrapolation error on
        # increasingly out-of-distribution actions (the buffer stays
        # dominated by older, less-confident policy data while the actor
        # sharpens), compounding as the policy shifts further from what the
        # critic was actually trained on. Every concentration-side patch so
        # far (ceiling, fixed penalty, adaptive penalty) fights this
        # DOWNSTREAM symptom; these two levers (both from Fujimoto et al.
        # 2018, TD3) attack the upstream cause directly, kept ON TOP of
        # diag16's dual-ascent penalty rather than replacing it:
        #   policy_delay: actor + target-network updates only run every
        #     `policy_delay` critic updates (critic updates every call, as
        #     always) -- gives the critic more gradient steps to correct
        #     itself before the actor can exploit its current errors.
        #   target_smoothing_*: the Bellman target's next-action is
        #     evaluated at a NOISED, re-projected-onto-the-simplex point
        #     instead of the actor's raw (possibly narrow/overconfident)
        #     sample -- adapts TD3's box-clipped Gaussian noise to a
        #     probability-simplex action via clamp+renormalize instead of
        #     clamp-to-range, so the target Q reflects a local neighborhood
        #     average rather than a single potentially-OOD point.
        # policy_delay=2 matches TD3's own paper default. smoothing sigma/
        # clip are scaled to ~30%/~75% of a "typical" simplex entry
        # (1/num_cols) rather than the [0,1] full range (TD3's own
        # proportions, applied to a full [-1,1] range, would be far too
        # large relative to how small one entry of a 15-way simplex row
        # typically is) -- first guesses, not yet measured, same as every
        # other coefficient in this file.
        self.policy_delay = policy_delay
        self.target_smoothing_sigma = target_smoothing_sigma
        self.target_smoothing_clip = target_smoothing_clip
        self._update_count = 0

        self.actor = Actor(state_dim, num_nodes, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    @property
    def alpha(self):
        return self.alpha_value

    def select_action(self, state, deterministic=False):
        """Returns (matrix[N,N+1], behavior_log_prob).

        DATASET (2026-09-29, offline-RL logging): behavior_log_prob is the
        log-density of the returned action under the exact distribution it
        was drawn from -- needed downstream for importance-sampling-style
        offline-RL corrections (CQL/BCQ and friends often want to know how
        likely the behavior policy was to take this action). None for
        deterministic calls (evaluation only, not used for data collection)
        since "log_prob of the mode" isn't the same well-defined quantity.
        Warmup's uniform-random Dirichlet(1,...,1) actions are NOT drawn
        through this method at all (see train.py's run_episode) -- their
        log_prob is left for a future reader to compute analytically from
        the known uniform-Dirichlet density if ever needed, rather than
        adding that here for a code path this method doesn't touch.
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                matrix = self.actor.deterministic(state_t)
                log_prob = None
            else:
                matrix, log_prob_t, _, _ = self.actor.sample(state_t)
                log_prob = float(log_prob_t.squeeze(0).item())
        return matrix.squeeze(0).cpu().numpy(), log_prob  # (N, N+1), float|None

    def update(self, batch):
        state, action, reward, next_state, done = (
            torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch
        )

        # ---- Critic update (Algorithm 1, step 32) ----
        with torch.no_grad():
            next_matrix, next_log_prob, _, _ = self.actor.sample(next_state)
            # BREAKTHROUGH (2026-09-28, diag17, see __init__'s comment):
            # TD3-style target policy smoothing, adapted to the simplex --
            # perturb the sampled next-action with small clipped noise, then
            # clamp to non-negative and renormalize each row back onto the
            # simplex (in place of TD3's clamp-to-box), so critic_target is
            # evaluated over a local neighborhood of the actor's next-action
            # rather than a single point it could learn to exploit.
            noise = (torch.randn_like(next_matrix) * self.target_smoothing_sigma).clamp(
                -self.target_smoothing_clip, self.target_smoothing_clip)
            next_matrix_smoothed = (next_matrix + noise).clamp(min=1e-8)
            next_matrix_smoothed = next_matrix_smoothed / next_matrix_smoothed.sum(dim=-1, keepdim=True)
            next_action_flat = next_matrix_smoothed.view(next_matrix_smoothed.size(0), -1)
            q1_next, q2_next = self.critic_target(next_state, next_action_flat)
            q_next = torch.min(q1_next, q2_next) - self.alpha_value * next_log_prob.unsqueeze(-1)
            target_q = reward + (1.0 - done) * self.gamma * q_next

        q1, q2 = self.critic(state, action)
        # FIX (2026-09-23, see Q1 report mục 12c/13): MSE squares the TD
        # error, so a single outlier transition (a slot where a host just
        # died -- drop_count can jump by hundreds) produces a huge gradient
        # that alone can yank the critic off-scale. Huber/smooth-L1 is
        # linear (not quadratic) beyond beta=1.0, bounding any one
        # transition's contribution regardless of how extreme -- standard
        # practice for reward-outlier robustness (DQN/Rainbow use the same
        # loss for the same reason). Combined with reward_scale (config.py)
        # and grad_clip_norm below.
        critic_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = 0.0
        if self.grad_clip_norm is not None:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.grad_clip_norm).item()
        self.critic_optim.step()

        # ---- Actor update (Algorithm 1, step 33 — entropy-constrained policy gradient) ----
        # BREAKTHROUGH (2026-09-28, diag17, see __init__'s comment): delayed
        # policy update (TD3) -- the critic above still updates every call;
        # the actor's gradient step, the conc_lambda dual-ascent step (tied
        # to the actor's own output), and the critic-target soft update are
        # all gated behind this and only run every `policy_delay` calls, so
        # the actor spends more of training reacting to an already-somewhat-
        # settled critic instead of chasing its most recent, noisiest
        # estimate. A no_grad forward pass still runs on skipped calls so
        # every field below stays populated every episode (train.py's
        # accumulators are unaware of the delay and expect a value each
        # call) -- only the actual parameter updates are skipped, not the
        # diagnostics.
        self._update_count += 1
        do_actor_update = (self._update_count % self.policy_delay == 0)

        if do_actor_update:
            matrix, log_prob, conc_diag, row_sum = self.actor.sample(state)
            action_flat = matrix.view(matrix.size(0), -1)
            q1_pi, q2_pi = self.critic(state, action_flat)
            q_pi = torch.min(q1_pi, q2_pi)
            # FIX (2026-09-27, diag15, see __init__'s comment): explicit,
            # gradient-carrying penalty on raw (pre-ceiling) concentration sum
            # exceeding conc_target -- zero when the row is already modest,
            # growing quadratically (a real restoring force, not a flat wall)
            # once it overshoots. row_sum shape is (batch, num_nodes, 1); mean
            # over both batch and node dimensions for a single scalar term.
            conc_penalty = (F.relu(row_sum.squeeze(-1) - self.conc_target) ** 2).mean()
            actor_loss = (self.alpha_value * log_prob.unsqueeze(-1) - q_pi).mean() \
                + self.conc_lambda * conc_penalty

            self.actor_optim.zero_grad()
            actor_loss.backward()
            actor_grad_norm = 0.0
            if self.grad_clip_norm is not None:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.grad_clip_norm).item()
            self.actor_optim.step()

            # ---- conc_lambda dual ascent (2026-09-28, diag16) ----
            # Plain dual-ascent step on the SIGNED mean violation (row_sum.mean()
            # - conc_target), reusing this step's own row_sum (no extra forward
            # pass): lambda grows when the actor is, on average, over target
            # (giving the penalty more pull next step) and relaxes when
            # comfortably under (letting the actor use its full concentration
            # budget again) -- a genuine two-sided controller, unlike alpha's
            # one-directional drift, because this constraint has a real
            # equilibrium (diag15 held row_sum ~17.0-17.24 for 80+ episodes
            # under a FIXED lambda; this replaces that guess with a value that
            # tracks whatever pull is actually needed as q_pi's own magnitude
            # grows over training). Deliberately NOT Adam-optimized machinery
            # (log-parameterized, its own optimizer, etc.) -- a plain clamped
            # additive step is auditable and low-risk, the same simplicity
            # bias every other coefficient in this file has converged to after
            # the log_alpha saga.
            with torch.no_grad():
                violation = row_sum.squeeze(-1).mean().item() - self.conc_target
                self.conc_lambda = min(max(
                    self.conc_lambda + self.conc_lambda_lr * violation,
                    self.conc_lambda_min), self.conc_lambda_max)

            # ---- Target network soft update (Algorithm 1, step 34) ----
            # Delayed together with the actor, matching TD3 Algorithm 1.
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        else:
            with torch.no_grad():
                matrix, log_prob, conc_diag, row_sum = self.actor.sample(state)
                action_flat = matrix.view(matrix.size(0), -1)
                q1_pi, q2_pi = self.critic(state, action_flat)
                q_pi = torch.min(q1_pi, q2_pi)
                conc_penalty = (F.relu(row_sum.squeeze(-1) - self.conc_target) ** 2).mean()
                actor_loss = (self.alpha_value * log_prob.unsqueeze(-1) - q_pi).mean() \
                    + self.conc_lambda * conc_penalty
            actor_grad_norm = 0.0

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha_value,
            "q_pi_mean": q_pi.mean().item(),
            "target_q_mean": target_q.mean().item(),
            "target_q_std": target_q.std().item(),
            "log_prob_mean": log_prob.mean().item(),
            "entropy_raw": self.alpha_value * log_prob.mean().item(),
            "entropy_bonus": self.alpha_value * log_prob.mean().item(),
            # FIX (2026-09-23): pre-clip grad norms, so grad_clip_norm
            # (config.py) can be re-derived empirically from real data next
            # time instead of guessed.
            "critic_grad_norm": critic_grad_norm,
            "actor_grad_norm": actor_grad_norm,
            # DIAGNOSTIC (2026-09-25, see networks.py comment + Q1 report mục 15,
            # 21): alpha0 = Dirichlet concentration SUM (statistics notation
            # alpha_0), NOT the SAC temperature above -- this is the quantity
            # diag13 showed actually drifting; watch it directly against the
            # new, tighter CONCENTRATION_SUM_CEILING (networks.py).
            "alpha0_mean": conc_diag["alpha0_mean"],
            "alpha0_max": conc_diag["alpha0_max"],
            "clip_frac": conc_diag["clip_frac"],
            "conc_entry_max": conc_diag["conc_entry_max"],
            # DIAGNOSTIC (2026-09-27, diag15): the raw penalty value (before
            # conc_lambda) and its weighted contribution to actor_loss --
            # watch conc_penalty_weighted against q_pi's own magnitude to
            # see whether conc_lambda is actually strong enough to matter,
            # the same "measure, don't guess" check every other coefficient
            # in this file has needed.
            "conc_penalty": conc_penalty.item(),
            "conc_penalty_weighted": (self.conc_lambda * conc_penalty).item(),
            # DIAGNOSTIC (2026-09-28, diag16): the CURRENT adapted value,
            # post-update -- watch this against alpha0_mean/q_pi_mean to
            # confirm it actually rises when q_pi's growing magnitude starts
            # to overpower a fixed lambda (the diag15 failure mode this is
            # meant to preempt), rather than sitting inert at its initial
            # guess.
            "conc_lambda": self.conc_lambda,
            # DIAGNOSTIC (2026-09-28, diag17): whether THIS call actually
            # ran the actor/target-network update (policy_delay gate) --
            # sanity-check that the schedule is firing as intended (should
            # average to 1/policy_delay across an episode's update() calls).
            "did_actor_update": float(do_actor_update),
        }

    def save(self, path, episode, norms=None):
        """Saves everything needed to resume training with pretrained weights.

        Java has no persistence of its own (DRLMainApp always restarts its
        episode counter at 1 -- see PROTOCOL.md section 4.4), so `episode`
        here is informational only (for logging), not something a future
        run can hand back to Java to skip ahead.

        norms (reward.RewardNorms, optional, 2026-09-26 reward redesign):
        the online mean/var trackers behind reward normalization -- saved
        alongside the network weights so a resume after crash/reboot picks
        up the same running statistics instead of restarting them from
        scratch (which would transiently distort the reward scale right
        after every resume).
        """
        ckpt = {
            "episode": episode,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            # ADAPTIVE (2026-09-28, diag16): conc_lambda is now learned state,
            # not a fixed hyperparameter -- persist it so a resume continues
            # from wherever the dual-ascent had settled instead of silently
            # resetting to config.py's initial guess.
            "conc_lambda": self.conc_lambda,
        }
        if norms is not None:
            ckpt["reward_norms"] = norms.state_dict()
        torch.save(ckpt, path)

    def load(self, path, norms=None):
        """Loads a checkpoint saved by save(). Returns the episode it was saved at.

        If norms is given and the checkpoint has a "reward_norms" entry
        (i.e. it was saved after the 2026-09-26 reward redesign), restores
        its state in place. A pre-redesign checkpoint simply has no such
        key -- norms then keeps its freshly-initialized state, which is the
        correct behaviour for a first-time upgrade. A checkpoint saved
        before diag14 (still carrying log_alpha/alpha_optim) simply has
        those extra keys ignored here.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        if norms is not None and "reward_norms" in ckpt:
            norms.load_state_dict(ckpt["reward_norms"])
        # ADAPTIVE (2026-09-28, diag16): a pre-diag16 checkpoint has no
        # "conc_lambda" key -- keeps self.conc_lambda at its freshly
        # constructed initial value (config.py's guess), the correct
        # behaviour for a first-time upgrade, same pattern as reward_norms.
        if "conc_lambda" in ckpt:
            self.conc_lambda = ckpt["conc_lambda"]
        return ckpt["episode"]
