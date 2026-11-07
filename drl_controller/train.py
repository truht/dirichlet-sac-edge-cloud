"""Entry point: Algorithm 1 -- Training & Decision-Making Procedure at
Central DRL Controller.

Connects to EdgeCloudSim (Java, TCP server) and runs Max_Episodes worth
of SAC training. See /PROTOCOL.md for the exact wire protocol.
"""

import dataclasses
import json
import os
import sys

import numpy as np

from config import Config
from protocol import EdgeCloudSimClient
from reward import RewardNorms, compute_reward, jain_fairness
from sac import ReplayBuffer, SACAgent

# stdout is fully block-buffered (not line-buffered) whenever it isn't a
# TTY -- e.g. `python3 train.py > log.txt` or piping through `tee`. Without
# this, per-episode progress lines sit in the buffer and don't show up in
# the log file / `tail -f` until the buffer fills or the process exits,
# which is exactly when you want to watch a long training run live.
sys.stdout.reconfigure(line_buffering=True)


def extract_energy(state, num_nodes):
    """E_i sits at offset 4 of each per-host 5-value block (see PROTOCOL.md section 2)."""
    return [state[i * 5 + 4] for i in range(num_nodes)]


def run_episode(client, cfg, agent, buffer, episode, msg, norms):
    """Runs one episode to completion and returns a stats dict for logging.

    The scalar reward is what SAC actually optimizes; cvar/fairness/drop/
    cloud-fraction are tracked separately so training can be diagnosed
    component-by-component instead of guessing from the combined reward
    curve alone (see the reward-shaping discussion this logging supports:
    a policy that always offloads to Cloud trivially minimizes the
    fairness and drop terms, so cloud_fraction climbing toward 1.0 is the
    signal to watch for a degenerate policy, not just rising return).
    """
    num_nodes = msg["num_nodes"]
    state = np.array(msg["state"], dtype=np.float32)
    # REDESIGN (2026-09-26, see reward.py module docstring + Q1 report mục 16):
    # captured once per episode so fairness can be computed on CUMULATIVE
    # DRAINED energy since episode start (energy_start - energy_now) instead
    # of raw remaining battery -- Jain's index on remaining battery stays
    # pinned at ~1.000 for hundreds of episodes whenever every host starts
    # near-full and drains slowly (the relative spread is tiny next to the
    # ~90-100% common baseline), so it carried ZERO usable variance in all
    # 18 finished runs regardless of device count. Drained-since-start grows
    # a real, comparable spread across hosts within a single ~108-slot
    # episode, well before absolute battery levels themselves ever diverge.
    energy_start = extract_energy(state, num_nodes)

    episode_return = 0.0
    cvar_sum = 0.0
    fairness_sum = 0.0
    drop_total = 0
    drop_dead_total = 0
    drop_vram_total = 0
    cloud_total = 0
    edge_total = 0
    time_slot = 0
    # Sum of the routing matrix across all slots -- lets us see how the
    # policy spreads load across the N *individual* Edge hosts, not just
    # the Edge-vs-Cloud split cloud_fraction already tracks. matrix[i][j]
    # is the probability host i routes to destination j (j==num_nodes is
    # Cloud), so summing column j over all source rows i (excluding the
    # Cloud column) gives the total probability mass received by Edge
    # host j -- the same idea as edge_routed_count but per-host instead
    # of pooled, and computed from the policy's own output so it needs no
    # changes on the Java side.
    matrix_sum = np.zeros((num_nodes, num_nodes + 1), dtype=np.float64)

    # DIAGNOSTIC (2026-09-22, ed-cd-diag1): accumulate agent.update()'s own
    # diagnostics (previously computed but discarded) so we can tell a
    # diverging/exploding critic apart from a critic that has converged
    # smoothly to a bad policy -- see mục 12 of the Q1 report. Averaged
    # over however many update() calls happen this episode (0 during the
    # warmup-only portion).
    update_count = 0
    critic_loss_sum = 0.0
    actor_loss_sum = 0.0
    q_pi_sum = 0.0
    target_q_sum = 0.0
    target_q_std_sum = 0.0
    log_prob_sum = 0.0
    entropy_bonus_sum = 0.0
    entropy_raw_sum = 0.0
    critic_grad_norm_sum = 0.0
    actor_grad_norm_sum = 0.0
    alpha0_mean_sum = 0.0
    alpha0_max_running = 0.0
    clip_frac_sum = 0.0
    conc_entry_max_running = 0.0
    conc_penalty_sum = 0.0
    conc_penalty_weighted_sum = 0.0
    did_actor_update_sum = 0.0
    # REDESIGN (2026-09-26): normalized reward-component sums, so the print
    # line shows what compute_reward actually combined (post-normalization),
    # not just the physical-unit cvar/fairness/drops that no longer directly
    # correspond to the reward's magnitude breakdown.
    norm_cvar_sum = 0.0
    norm_fair_sum = 0.0
    norm_drop_sum = 0.0

    # Opt-in raw transition capture (see Config.save_transitions) -- kept
    # as plain Python lists during the episode and stacked into arrays only
    # at the end, so the per-step loop below costs nothing extra when the
    # flag is off (these stay empty and unused).
    trans_states, trans_actions, trans_rewards = [], [], []
    trans_next_states, trans_dones = [], []
    # DATASET (2026-09-29): the "_meta" side of the split -- see config.py's
    # save_transitions comment for why these live in a separate file from
    # the 5 lines above instead of alongside them.
    trans_behavior_log_probs, trans_behavior_types = [], []
    trans_cvar_raw, trans_fairness_raw, trans_drop_raw = [], [], []

    while True:
        if len(buffer) >= cfg.warmup_steps:
            # FIX (2026-09-27, see reward.py's EMANorm comment): freeze the
            # reward normalizers' mean/var right as random actions give way
            # to the learned policy -- idempotent, so this is a no-op on
            # every step after the first. Locks the reference frame to
            # genuinely-random behavior before the policy being judged by
            # it ever gets a chance to drag it along.
            norms.freeze_all()
            matrix, behavior_log_prob = agent.select_action(state)
            behavior_type = cfg.BEHAVIOR_LEARNED_POLICY
        else:
            # Uniform-random valid action (Dirichlet(1,...,1) is uniform
            # over the simplex) until the buffer holds enough transitions
            # for meaningful gradient updates.
            matrix = np.random.dirichlet(np.ones(num_nodes + 1), size=num_nodes)
            # DATASET (2026-09-29): not computed here (see agent.py's
            # select_action docstring) -- a future reader who needs it can
            # derive it analytically, since this is exactly the KNOWN
            # uniform Dirichlet(1,...,1) density, unlike the learned
            # policy's state-dependent one.
            behavior_log_prob = float("nan")
            behavior_type = cfg.BEHAVIOR_RANDOM_WARMUP

        time_slot += 1
        matrix_sum += matrix
        client.send_action(episode, time_slot, matrix.tolist())

        msg = client.recv_message()
        if msg["type"] != "transition":
            raise RuntimeError(f"Expected transition, got {msg['type']}")

        next_state = np.array(msg["next_state"], dtype=np.float32)
        energy_now = extract_energy(next_state, num_nodes)
        # REDESIGN (2026-09-26): drained-since-episode-start, not raw
        # remaining battery -- see the energy_start comment above.
        drained = [s - n for s, n in zip(energy_start, energy_now)]
        reward, cvar, fairness, norm_cvar, norm_inv_fair, norm_drop = compute_reward(
            msg["completed_latencies_ms"], drained, msg["drop_count"],
            cfg.cvar_beta, cfg.w1, cfg.w2, cfg.w3, norms,
        )
        # FIX (2026-09-23, see Q1 report mục 13): scale the reward fed to
        # SAC (buffer/return) so Q-value magnitude is well-conditioned for a
        # 256-unit critic -- cvar/fairness/drop_total logging below stays in
        # original physical units (unaffected), only the RL reward signal
        # is scaled.
        reward *= cfg.reward_scale
        episode_return += reward
        cvar_sum += cvar
        fairness_sum += fairness
        norm_cvar_sum += norm_cvar
        norm_fair_sum += norm_inv_fair
        norm_drop_sum += norm_drop
        drop_total += msg["drop_count"]
        drop_dead_total += msg["drop_dead_count"]
        drop_vram_total += msg["drop_vram_count"]
        cloud_total += msg["cloud_routed_count"]
        edge_total += msg["edge_routed_count"]
        done = msg["done"]

        buffer.add(state, matrix.flatten(), reward, next_state, done)

        if cfg.save_transitions:
            trans_states.append(state)
            trans_actions.append(matrix.flatten())
            trans_rewards.append(reward)
            trans_next_states.append(next_state)
            trans_dones.append(done)
            trans_behavior_log_probs.append(behavior_log_prob)
            trans_behavior_types.append(behavior_type)
            # cvar/fairness are already RAW physical-unit values here (see
            # compute_reward's own docstring) -- msg["drop_count"] likewise
            # raw, straight off the wire. None of the three are the
            # normalized norm_cvar/norm_inv_fair/norm_drop that actually
            # went into the reward SAC optimizes.
            trans_cvar_raw.append(cvar)
            trans_fairness_raw.append(fairness)
            trans_drop_raw.append(msg["drop_count"])

        if len(buffer) >= cfg.warmup_steps:
            upd = agent.update(buffer.sample(cfg.batch_size))
            update_count += 1
            critic_loss_sum += upd["critic_loss"]
            actor_loss_sum += upd["actor_loss"]
            q_pi_sum += upd["q_pi_mean"]
            target_q_sum += upd["target_q_mean"]
            target_q_std_sum += upd["target_q_std"]
            log_prob_sum += upd["log_prob_mean"]
            entropy_bonus_sum += upd["entropy_bonus"]
            entropy_raw_sum += upd["entropy_raw"]
            critic_grad_norm_sum += upd["critic_grad_norm"]
            actor_grad_norm_sum += upd["actor_grad_norm"]
            alpha0_mean_sum += upd["alpha0_mean"]
            alpha0_max_running = max(alpha0_max_running, upd["alpha0_max"])
            clip_frac_sum += upd["clip_frac"]
            conc_entry_max_running = max(conc_entry_max_running, upd["conc_entry_max"])
            conc_penalty_sum += upd["conc_penalty"]
            conc_penalty_weighted_sum += upd["conc_penalty_weighted"]
            did_actor_update_sum += upd["did_actor_update"]

        state = next_state
        if done:
            break

    if cfg.save_transitions:
        # DATASET (2026-09-29): core file -- exactly the standard 5-key
        # offline-RL tuple, unchanged shape from before this change. Any
        # generic/off-the-shelf offline-RL loader only ever needs this file.
        core_path = os.path.join(cfg.transitions_dir, f"ep_{episode:05d}.npz")
        np.savez_compressed(
            core_path,
            states=np.stack(trans_states),
            actions=np.stack(trans_actions),
            rewards=np.array(trans_rewards, dtype=np.float32),
            next_states=np.stack(trans_next_states),
            dones=np.array(trans_dones, dtype=bool),
        )
        # DATASET (2026-09-29): meta file -- this project's research-specific
        # extras, deliberately kept out of the core file (see config.py's
        # save_transitions comment for the full rationale). Written in the
        # same block as core_path above so the two can never go out of sync.
        meta_path = os.path.join(cfg.transitions_dir, f"ep_{episode:05d}_meta.npz")
        np.savez_compressed(
            meta_path,
            behavior_log_probs=np.array(trans_behavior_log_probs, dtype=np.float32),
            behavior_types=np.array(trans_behavior_types, dtype=np.int8),
            cvar_raw=np.array(trans_cvar_raw, dtype=np.float32),
            fairness_raw=np.array(trans_fairness_raw, dtype=np.float32),
            drop_raw=np.array(trans_drop_raw, dtype=np.int32),
        )

    routed_total = cloud_total + edge_total

    # edge_incoming[j] = total probability mass routed to Edge host j across
    # the whole episode, summed over every source host i. jain_fairness on
    # this vector is 1.0 only if every Edge host receives an equal share --
    # exactly the "phân tải trên nhiều edge" question, separate from the
    # existing energy-based avg_fairness (which measures battery balance,
    # not routing balance) and from cloud_fraction (Edge-vs-Cloud only,
    # blind to imbalance *within* the Edge tier).
    edge_incoming = matrix_sum[:, :num_nodes].sum(axis=0)
    edge_total_mass = edge_incoming.sum()
    if edge_total_mass > 1e-9:
        edge_balance = jain_fairness(edge_incoming)
        edge_share_min = float(edge_incoming.min() / edge_total_mass)
        edge_share_max = float(edge_incoming.max() / edge_total_mass)
    else:
        edge_balance = 1.0
        edge_share_min = 0.0
        edge_share_max = 0.0

    n = max(update_count, 1)  # avoid div-by-zero on warmup-only episodes (update_count==0)
    return {
        "slots": time_slot,
        "return": episode_return,
        "avg_cvar_ms": cvar_sum / time_slot,
        "avg_fairness": fairness_sum / time_slot,
        "drop_total": drop_total,
        "drop_dead_total": drop_dead_total,
        "drop_vram_total": drop_vram_total,
        "cloud_fraction": (cloud_total / routed_total) if routed_total > 0 else 0.0,
        "edge_balance": edge_balance,
        "edge_share_min": edge_share_min,
        "edge_share_max": edge_share_max,
        "update_count": update_count,
        "critic_loss": critic_loss_sum / n,
        "actor_loss": actor_loss_sum / n,
        "q_pi_mean": q_pi_sum / n,
        "target_q_mean": target_q_sum / n,
        "target_q_std": target_q_std_sum / n,
        "log_prob_mean": log_prob_sum / n,
        "entropy_bonus": entropy_bonus_sum / n,
        "entropy_raw": entropy_raw_sum / n,
        "critic_grad_norm": critic_grad_norm_sum / n,
        "actor_grad_norm": actor_grad_norm_sum / n,
        "alpha0_mean": alpha0_mean_sum / n,
        "alpha0_max": alpha0_max_running,
        "clip_frac": clip_frac_sum / n,
        "conc_entry_max": conc_entry_max_running,
        "conc_penalty": conc_penalty_sum / n,
        "conc_penalty_weighted": conc_penalty_weighted_sum / n,
        "did_actor_update_frac": did_actor_update_sum / n,
        "norm_cvar": norm_cvar_sum / time_slot,
        "norm_fair": norm_fair_sum / time_slot,
        "norm_drop": norm_drop_sum / time_slot,
    }


def write_manifest(cfg, agent, norms, num_nodes, state_dim, last_episode):
    """DATASET (2026-09-29): one manifest.json per run, written once at the
    very end (see main()'s finally block) -- everything a future reader
    needs to interpret ep_NNNNN.npz / ep_NNNNN_meta.npz WITHOUT access to
    this conversation or the code that produced them: full config, the
    state/action schema (authoritative source: DRLEdgeOrchestrator.
    buildStateVector, see PROTOCOL.md section 2), and RewardNorms' frozen
    mean/var (to de-normalize norm_cvar/norm_inv_fair/norm_drop back to
    physical units from just a checkpoint, without this run's raw log).
    """
    manifest = {
        "config": dataclasses.asdict(cfg),
        "num_nodes": num_nodes,
        "state_dim": state_dim,
        "action_dim": num_nodes * (num_nodes + 1),
        "last_episode": last_episode,
        "state_schema": (
            "state[i*5 + 0] = queue length of edge host i (running + waiting cloudlets); "
            "state[i*5 + 1] = 100 - CPU used% (i.e. free CPU %); "
            "state[i*5 + 2] = 100 - GPU used% (free GPU %); "
            "state[i*5 + 3] = free RAM of host i, GB; "
            "state[i*5 + 4] = battery remaining % of host i; "
            "for i in [0, num_nodes); state[-1] = time_slot / max_slots_this_episode "
            "(normalized episode progress, the only non-per-host entry). "
            "Authoritative source: DRLEdgeOrchestrator.buildStateVector (Java)."
        ),
        "action_schema": (
            "action reshapes to (num_nodes, num_nodes+1) row-major; row i is a "
            "probability simplex (sums to 1, non-negative) over where host i's "
            "incoming load this slot is routed: columns [0, num_nodes) are the "
            "other edge hosts, column num_nodes (the last one) is Cloud."
        ),
        "behavior_type_legend": {
            str(cfg.BEHAVIOR_RANDOM_WARMUP): "random_warmup (uniform Dirichlet(1,...,1); behavior_log_prob is NaN -- see meta file)",
            str(cfg.BEHAVIOR_LEARNED_POLICY): "learned_policy (agent.select_action, stochastic sample; behavior_log_prob is the exact log-density)",
        },
        "reward_norms_frozen": norms.state_dict(),
        "reward_formula": "reward = -(w1*norm_cvar + w2*norm_inv_fair + w3*norm_drop), then scaled by reward_scale before going into the replay buffer / ep_NNNNN.npz's rewards; ep_NNNNN_meta.npz's cvar_raw/fairness_raw/drop_raw are the PRE-normalization physical-unit values compute_reward() computed norm_cvar/norm_inv_fair/norm_drop from.",
    }
    path = os.path.join(cfg.transitions_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[drl_controller] wrote dataset manifest to {path}")


def main():
    cfg = Config()
    client = EdgeCloudSimClient(cfg.host, cfg.port)
    print(f"[drl_controller] connected to EdgeCloudSim at {cfg.host}:{cfg.port}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
    if cfg.save_transitions:
        os.makedirs(cfg.transitions_dir, exist_ok=True)
        print(f"[drl_controller] saving raw transitions per episode to {cfg.transitions_dir}/")

    agent = None
    buffer = None
    episode = 0
    # REDESIGN (2026-09-26): one RewardNorms per training run, independent of
    # num_nodes/state_dim so it can be constructed before the first
    # episode_start (agent itself still waits for that, since its network
    # sizes do depend on them).
    norms = RewardNorms()

    try:
        for episode in range(1, cfg.max_episodes + 1):
            msg = client.recv_message()
            if msg["type"] != "episode_start":
                raise RuntimeError(f"Expected episode_start, got {msg['type']}")

            if agent is None:
                num_nodes = msg["num_nodes"]
                state_dim = msg["state_dim"]
                agent = SACAgent(state_dim, num_nodes, hidden_dim=cfg.hidden_dim,
                                  gamma=cfg.gamma, tau=cfg.tau, lr=cfg.lr, device=cfg.device,
                                  critic_lr=cfg.critic_lr, grad_clip_norm=cfg.grad_clip_norm,
                                  alpha_fixed=cfg.alpha_fixed,
                                  conc_target=cfg.conc_target, conc_lambda=cfg.conc_lambda,
                                  conc_lambda_min=cfg.conc_lambda_min,
                                  conc_lambda_max=cfg.conc_lambda_max,
                                  conc_lambda_lr=cfg.conc_lambda_lr,
                                  policy_delay=cfg.policy_delay,
                                  target_smoothing_sigma=cfg.target_smoothing_sigma,
                                  target_smoothing_clip=cfg.target_smoothing_clip)
                buffer = ReplayBuffer(cfg.buffer_capacity, state_dim, num_nodes * (num_nodes + 1))

                if os.path.exists(checkpoint_path):
                    # Only the network/optimizer weights resume -- the replay
                    # buffer starts empty and Java's own episode counter
                    # restarts at 1 regardless (see Config.checkpoint_dir doc).
                    loaded_episode = agent.load(checkpoint_path, norms=norms)
                    print(f"[drl_controller] resumed weights from {checkpoint_path} "
                          f"(saved at episode {loaded_episode}); replay buffer starts empty")

            stats = run_episode(client, cfg, agent, buffer, episode, msg, norms)
            print(f"[episode {episode:4d}] slots={stats['slots']:4d} "
                  f"return={stats['return']:10.3f} "
                  f"avg_cvar={stats['avg_cvar_ms']:8.2f}ms "
                  f"avg_fairness={stats['avg_fairness']:.3f} "
                  f"drops={stats['drop_total']:5d}(dead={stats['drop_dead_total']:5d},vram={stats['drop_vram_total']:5d}) "
                  f"cloud%={stats['cloud_fraction'] * 100:5.1f} "
                  f"edge_balance={stats['edge_balance']:.3f} "
                  f"edge_share=[{stats['edge_share_min']*100:4.1f}%,{stats['edge_share_max']*100:4.1f}%] "
                  f"buffer={len(buffer)} alpha={agent.alpha:.4f} "
                  f"critic_loss={stats['critic_loss']:12.2f} "
                  f"actor_loss={stats['actor_loss']:10.3f} "
                  f"q_pi={stats['q_pi_mean']:10.2f} "
                  f"target_q={stats['target_q_mean']:10.2f}(sd={stats['target_q_std']:8.2f}) "
                  f"log_prob={stats['log_prob_mean']:7.3f} "
                  f"entropy_bonus={stats['entropy_bonus']:7.3f} "
                  f"entropy_raw={stats['entropy_raw']:9.3f} "
                  f"critic_grad={stats['critic_grad_norm']:8.3f} "
                  f"actor_grad={stats['actor_grad_norm']:8.3f} "
                  f"alpha0={stats['alpha0_mean']:6.2f}(max={stats['alpha0_max']:6.2f}) "
                  f"clip_frac={stats['clip_frac']:.3f} "
                  f"conc_pen={stats['conc_penalty']:8.3f}(w={stats['conc_penalty_weighted']:7.4f}) "
                  f"conc_lambda={agent.conc_lambda:.5f} "
                  f"actor_upd_frac={stats['did_actor_update_frac']:.3f} "
                  f"conc_max={stats['conc_entry_max']:6.2f} "
                  f"norm_cvar={stats['norm_cvar']:7.3f} "
                  f"norm_fair={stats['norm_fair']:7.3f} "
                  f"norm_drop={stats['norm_drop']:7.3f} "
                  f"upd={stats['update_count']:3d}")

            if episode % cfg.checkpoint_every == 0:
                agent.save(checkpoint_path, episode, norms=norms)
    except KeyboardInterrupt:
        print("[drl_controller] interrupted, sending early-stop shutdown")
        client.send_shutdown()
    finally:
        if agent is not None:
            agent.save(checkpoint_path, episode, norms=norms)
            print(f"[drl_controller] saved checkpoint to {checkpoint_path} (episode {episode})")
            if cfg.save_transitions:
                write_manifest(cfg, agent, norms, num_nodes, state_dim, episode)
        client.close()

    print("[drl_controller] training finished")


if __name__ == "__main__":
    main()
