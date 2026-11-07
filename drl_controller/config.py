"""Hyperparameters for the Central DRL Coordinator (Algorithm 1).

Kept in a plain dataclass so every value can be overridden from a single
place. Max_Episodes is NOT sent over the socket (see /PROTOCOL.md section
4.4) -- it must be kept in sync with whatever the Java side's own config
uses.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Socket (Java is the server, this process is the client).
    # NOTE: port 5000 conflicts with macOS ControlCenter/AirPlay Receiver,
    # which binds it by default -- verified during smoke testing. 6000
    # avoids that collision; keep in sync with DRLMainApp.SOCKET_PORT.
    host: str = "localhost"
    port: int = 7421

    # SAC hyperparameters
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    # FIX (2026-09-23, ed-cd-diag2 -- see Q1 report mục 12/13): diag1 showed
    # critic_loss/q_pi growing ~unboundedly from episode ~10 onward (1.7k ->
    # 320k by ep91), tracking real behavior collapsing in lockstep (fairness
    # 0.90->0.42, dead-drops 0->24k) -- classic off-policy Q-overestimation
    # spiral, not convergence to a genuinely-bad-but-stable policy (ablation
    # A vs B, mục 12b, already ruled out "reward under-penalizes drops").
    # critic_lr lower than actor/alpha's `lr` so the (currently unstable)
    # critic's errors propagate into the policy more slowly while the other
    # fixes below take hold.
    critic_lr: float = 1e-4
    # Bounds the per-update parameter step regardless of loss scale --
    # independent safety net alongside reward_scale/huber below. No prior
    # gradient-norm data existed to calibrate this exactly (train.py now
    # logs critic_grad_norm/actor_grad_norm so future runs can retune it);
    # 10.0 is a standard conservative default for continuous-control SAC/TD3.
    grad_clip_norm: float = 10.0
    buffer_capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1_000  # random actions until the buffer has this many transitions
    device: str = "cpu"  # set to "cuda" if a GPU is available

    # FIX (2026-09-23): raw per-slot reward (return/108) empirically measured
    # across ed-cd-A1..A6 (n=6, 2400 episode-samples): healthy/random-policy
    # phase ~-130 (std~1.3), full range across the run [-550,-130], mean
    # -359 (std 107) -- i.e. reward was NEVER normalized/scaled, so Q (a
    # discounted sum over up to 108 steps at gamma=0.99, discount factor
    # (1-0.99^108)/0.01~=66) naturally lands at -sum in the thousands to tens
    # of thousands purely from reward scale, before any overestimation bias.
    # Scaling by 0.01 maps the observed [-550,-130] range to
    # [-5.5,-1.3] -- a well-conditioned magnitude for a 256-unit MLP critic
    # with lr=1e-4/3e-4 (matches the "reward scale" hyperparameter the
    # original SAC paper (Haarnoja 2018) tunes per-environment; never done
    # here before). Applied to the scalar reward only, AFTER compute_reward
    # -- avg_cvar_ms/avg_fairness/drop_total logging stays in original
    # physical units for interpretability.
    reward_scale: float = 0.01

    # FIX round 2 (2026-09-23, see Q1 report mục 14): the FIRST widening
    # (log_alpha_max=5, i.e. alpha<=148.4) was itself a bug, not a fix --
    # ALL 12 production runs (ed-sac-300/900-s1..s6) AND diag2 hit this
    # exact ceiling (alpha=148.4132), because log_prob sits persistently
    # (if only by ~0.01-0.02) ABOVE -target_entropy every single update --
    # entropy never quite reaches the target, so alpha_loss's gradient sign
    # never flips, and Adam's near-constant-sign step size makes log_alpha
    # drift ~linearly (~0.03/episode) with NO natural equilibrium, only
    # whatever ceiling is set. (This reframes the ORIGINAL clamp too: it
    # was never a calibrated equilibrium, just an earlier interception of
    # the same one-directional drift.) log_alpha_max now set from a real
    # calculation instead of guessed: keep entropy_bonus (~alpha*log_prob,
    # log_prob observed ~25.2 stable) under ~10% of a healthy-regime |q_pi|
    # (~86, from reward_scale=0.01 * discount-sum factor ~66) ->
    # alpha <= 0.1*86/25.2 ~= 0.34 -> log_alpha_max ~= -1.1. log_alpha_min
    # left wide (drift is always upward in every run observed, never down,
    # so the lower bound has caused no issue).
    # REMOVED (2026-09-27, diag14, see sac/agent.py module docstring + Q1
    # report mục 21): automatic entropy temperature tuning (log_alpha_min/
    # max, target_entropy_scale) dropped entirely. diag13 showed the SAC
    # temperature itself was NOT what drifted -- it sat perfectly pinned at
    # its clamp (0.0224) for 150+ consecutive episodes; what actually
    # drifted was the actor's own Dirichlet concentration output (a
    # distinct quantity -- see networks.py's CONCENTRATION_SUM_CEILING,
    # now tightened as the real, targeted fix). With alpha already
    # effectively constant in practice, the auto-tuning machinery was
    # dead weight -- replaced with a genuinely fixed value instead of kept
    # as unused complexity.
    alpha_fixed: float = 0.0224
    # ADDED (2026-09-27, diag15, see sac/agent.py __init__ comment): explicit
    # gradient-carrying penalty on raw concentration sum exceeding
    # conc_target, on top of the forward-only CONCENTRATION_SUM_CEILING
    # (networks.py) -- diag14 showed the ceiling alone lets alpha0 climb
    # unbounded (16->54) behind a permanently saturated clip_frac=1.0 with no
    # gradient cost, eventually degrading drop_total anyway (~ep100+).
    conc_target: float = 17.0
    # ADAPTIVE (2026-09-28, diag16, see sac/agent.py's dual-ascent comment):
    # diag15 (fixed conc_lambda=0.01) held alpha0 near-flat through ep~80,
    # but q_pi's own magnitude was simultaneously growing fast (~-12.9 ep53
    # -> ~-19.4 ep83, +50% in 30 episodes, still climbing toward the
    # ~86-magnitude "fully converged" regime this reward_scale implies --
    # see reward_scale's own comment above) while conc_lambda stayed fixed --
    # i.e. the -q_pi term's pull keeps growing but the penalty's counter-pull
    # does not, so a FIXED value that balances today is not guaranteed to
    # still balance once q_pi reaches its natural full magnitude. Unlike the
    # SAC-alpha dual-ascent removed in diag14 (target_entropy had NO natural
    # equilibrium -- log_prob sat persistently above it, so log_alpha only
    # ever drifted one direction), THIS constraint (row_sum ~= conc_target)
    # has a real, observed two-sided equilibrium (diag15 held ~17.0-17.24
    # for 80+ episodes) -- the same dual-ascent MACHINERY that failed for
    # alpha should work correctly here because the underlying constraint is
    # actually satisfiable. conc_lambda below is now only the INITIAL value;
    # sac/agent.py updates it every training step via
    # conc_lambda += conc_lambda_lr * (row_sum.mean() - conc_target),
    # clamped to [conc_lambda_min, conc_lambda_max] as a hard safety net
    # (never trust an auto-tuned quantity fully unbounded -- the exact
    # lesson diag9-13 already paid for once with log_alpha). All three
    # *_lr/*_min/*_max values are first guesses, not yet measured -- watch
    # the new conc_lambda print field against alpha0/q_pi to see whether it
    # tracks the growing violation as intended.
    conc_lambda: float = 0.01
    conc_lambda_min: float = 0.001
    conc_lambda_max: float = 1.0
    conc_lambda_lr: float = 0.001

    # BREAKTHROUGH (2026-09-28, diag17, see sac/agent.py's __init__ comment):
    # diag15's own log data showed q_pi (~-33 at ep180) running ~20-30x
    # larger than what the actually-observed episode returns (~-1 to -3)
    # could support for an unbiased critic -- not legitimate convergence,
    # but off-policy extrapolation-error overestimation (Fujimoto et al.
    # 2018), which then keeps pushing the actor toward higher-confidence
    # (higher-concentration) actions regardless of how well-tuned the
    # concentration-penalty side is. Two standard TD3 levers, attacking
    # this upstream instead of another concentration-side patch:
    policy_delay: int = 2            # actor + target-net update every Nth critic update (TD3 paper default)
    target_smoothing_sigma: float = 0.02   # noise std added to the target next-action before Q(s',a') (~30% of a typical 1/num_cols simplex entry)
    target_smoothing_clip: float = 0.05    # hard clip on that noise (~75% of a typical entry)

    # Checkpointing -- Java has no persistence of its own (restarts its episode
    # counter at 1 every run, see PROTOCOL.md 4.4), so this only carries the
    # SAC agent's learned weights across separate train.py invocations, not
    # the episode count itself.
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 2  # episodes -- tight, data-safety practice for a cloud ablation batch

    # Transition-level logging (2026-09-29, offline-RL dataset -- see Q1
    # report mục 23): the per-episode print in train.py covers cross-run
    # comparison, but has no per-step (state, action, reward, next_state)
    # record -- the replay buffer holding those lives only in RAM and is
    # discarded every restart. Turned ON for diag19+ / the eventual
    # production runs, specifically to produce a reusable offline-RL
    # dataset (train a different algorithm on the same experience, relearn
    # w1/w2/w3, etc.) WITHOUT needing to re-run the simulation later.
    #
    # TWO files per episode, deliberately split (discussed at length before
    # implementing -- see conversation, not reproduced here):
    #   ep_NNNNN.npz      -- the standard 5-key offline-RL tuple (states,
    #                        actions, rewards, next_states, dones), the
    #                        ONLY file a generic/off-the-shelf offline-RL
    #                        loader needs to touch. Unaffected by anything
    #                        below -- kept byte-for-byte the same shape as
    #                        before this change.
    #                        ~108 steps x (71-dim state + 210-dim action +
    #                        71-dim next_state + reward + done)/episode.
    #   ep_NNNNN_meta.npz -- this project's own research-specific extras,
    #                        kept OUT of the core file so a standard loader
    #                        never has to know about them: behavior_log_probs
    #                        (None -> NaN for warmup's random actions, which
    #                        don't go through agent.select_action at all),
    #                        behavior_types (0=random_warmup, 1=learned_policy,
    #                        see BEHAVIOR_* below), and the raw (pre-
    #                        normalization) reward components (cvar_ms,
    #                        fairness, drop_count) -- compute_reward() only
    #                        ever returns the combined normalized scalar to
    #                        the replay buffer, so these would otherwise be
    #                        lost forever once written, closing off re-
    #                        deriving w1/w2/w3 later without a full re-run.
    # Both files are written in the same save_transitions block (see
    # run_episode's end) so they can never go out of sync (one written,
    # the other missing).
    #
    # Manifest -- ONE <transitions_dir>/manifest.json per run (not per
    # episode), written once at the very end (see main()'s finally block,
    # after norms are certainly frozen and the run's true final episode
    # count is known): full config dump, the state/action schema (which
    # index means what -- see PROTOCOL.md section 2 / DRLEdgeOrchestrator.
    # buildStateVector for the authoritative source), and RewardNorms'
    # frozen mean/var (needed to de-normalize norm_cvar/norm_fair/norm_drop
    # back to physical units if a future reader only has the checkpoint's
    # frozen norms and not this run's raw components).
    save_transitions: bool = True
    transitions_dir: str = "transitions"
    BEHAVIOR_RANDOM_WARMUP: int = 0
    BEHAVIOR_LEARNED_POLICY: int = 1

    # Reward (Algorithm 1, step 27)
    cvar_beta: float = 0.90
    # REDESIGN (2026-09-26, see reward.py module docstring + Q1 report mục
    # 16): superseded the old hand-picked raw-scale weights below. Recomputed
    # directly from all 18 finished SAC-B logs (seed1, full 400 episodes):
    # CVaR ate 97.5-99.3% of the raw combined-reward magnitude at EVERY
    # device level (300/600/900) despite varying by only 0.34-1.04% relative
    # SD -- essentially a large, near-constant offset, not a learnable
    # signal -- while drop_count (the only component with real,
    # policy-relevant variance: 30.6% relative SD at 900, 61.2% at 600) was
    # squeezed under 1.7% of the magnitude. Matching a fixed w3 to a target
    # share needed ~38 at 900 but ~1095 at 600 (>25x apart) -- no single
    # fixed weight generalizes across device levels. compute_reward now
    # z-score-normalizes each raw component online (reward.EMANorm) BEFORE
    # applying w1/w2/w3, so these three are post-normalization EMPHASIS
    # knobs (each component contributes comparable variance at 1:1:1) rather
    # than magnitude-compensation hacks that must be re-derived per device
    # count. Old raw-scale history (w1 0.01->0.1->0.5, w2=w3=1.0) kept only
    # in git-blame/report mục 12-13, not reproduced here -- it no longer
    # means the same thing under normalization.
    w1: float = 1.0
    w2: float = 1.0
    w3: float = 1.0

    # Training loop length -- must match the Java side's own episode count
    # (DRLMainApp.MAX_EPISODES). Raised 10x (500 -> 5000, ~540k steps) after
    # a 500-episode / 54k-step run showed flat return regardless of
    # target_entropy_scale -- testing whether more steps is the actual
    # bottleneck now that alpha's runaway-growth bug is fixed.
    # CHARGE-DISCHARGE ABLATION (2026-08-29, cloud): 400 matches the original
    # w2/w3 ablation's protocol for direct comparability (see
    # experiments/2026-08-29_w2w3_ablation/README.md) -- the local Mac smoke
    # test showed the first battery death around ep51 and clear recovery by
    # ep98 under these same energy_model parameters, so 400 leaves ~350
    # episodes of post-first-death data per run, plenty for a last-50-episode-
    # average comparison.
    max_episodes: int = 400
