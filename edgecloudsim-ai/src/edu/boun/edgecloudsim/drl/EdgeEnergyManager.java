/*
 * Title:        EdgeCloudSim - EdgeEnergyManager
 *
 * Description:
 * Per-host battery/energy model used to compute E_i^t for the DRL
 * state vector and reward (Jain's Fairness term). Tracks a remaining
 * energy budget per Edge Host that depletes over time as a function of
 * utilization, resets to full at the start of every SAC episode, and
 * flags a host as "dead" (Constraint C5-style hard failure) once its
 * budget reaches zero.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.drl;

/**
 * Tracks per-host remaining battery energy across a SAC episode.
 *
 * <p>Power budget is read per-host from <power_w> in edge_devices.xml (via
 * {@code DualChannelEdgeVm.getPowerW()}, see DRLEdgeOrchestrator's
 * buildPowerBudgetPerHost()) rather than from a single shared value, so
 * hosts with different hardware profiles can drain at different rates and
 * the Jain's Fairness term across hosts stays meaningful.</p>
 *
 * <p>Power model per time-slot:
 * <pre>
 *   draw(t)   = P_idle + (P_max - P_idle) * utilization(t)
 *   E(t+1)    = max(0, E(t) - draw(t) * deltaSeconds)
 * </pre>
 * Battery capacity is derived from each host's own power budget and the
 * episode duration so that a host sustained near max utilization for a
 * full episode drains close to empty, giving the SAC policy a real
 * load-balancing incentive instead of a budget that is either trivially
 * full or trivially empty.</p>
 */
public class EdgeEnergyManager {

    /**
     * Idle draw as a fraction of max power budget — servers never draw 0W at rest.
     *
     * <p>Lowered from 0.25 to 0.10 after real training data (300 and 1500 devices)
     * showed Jain's Fairness essentially frozen at 1.000 regardless of routing
     * skew: at 0.25, the load-INDEPENDENT idle draw dominates total energy use,
     * so the load-DEPENDENT term (the only part routing decisions can influence)
     * barely moves E_i relative to the host-to-host idle baseline they all share
     * equally. 5x more devices (300->1500) only nudged fairness to 0.999 -- proof
     * the bottleneck was this ratio, not sample size. 0.10 also better matches a
     * real Jetson TX2's idle draw (~7-13% of its ~15W TDP), so this is a modeling
     * correction, not just an RL-sensitivity tweak.</p>
     */
    private static final double IDLE_POWER_FRACTION = 0.10;

    /** Battery capacity = maxPowerW * episodeDurationSec * this factor. */
    private static final double DEPLETION_FACTOR = 0.9;

    private final int numberOfHost;
    private final double[] maxPowerW;
    private final double[] idlePowerW;
    private final double[] capacityJ;
    private final double[] remainingJ;

    /**
     * @param maxPowerWPerHost   power budget in Watts for each host, index-aligned
     *                           (from DualChannelEdgeVm.getPowerW(), see
     *                           DRLEdgeOrchestrator.buildPowerBudgetPerHost())
     * @param episodeDurationSec total simulated seconds per episode (T_max * deltaSeconds)
     */
    public EdgeEnergyManager(double[] maxPowerWPerHost, double episodeDurationSec) {
        this.numberOfHost = maxPowerWPerHost.length;
        this.maxPowerW = maxPowerWPerHost.clone();
        this.idlePowerW = new double[numberOfHost];
        this.capacityJ = new double[numberOfHost];
        this.remainingJ = new double[numberOfHost];
        for (int i = 0; i < numberOfHost; i++) {
            idlePowerW[i] = maxPowerW[i] * IDLE_POWER_FRACTION;
            capacityJ[i] = maxPowerW[i] * episodeDurationSec * DEPLETION_FACTOR;
        }
        resetAllFull();
    }

    /** Reset every host's battery to full capacity. Call at the start of every episode. */
    public void resetAllFull() {
        System.arraycopy(capacityJ, 0, remainingJ, 0, numberOfHost);
    }

    /**
     * Advance host {@code hostId}'s battery by one time-slot.
     *
     * @param hostId         0-based Edge Host index
     * @param utilizationPct combined utilization of that host in this slot, 0-100
     * @param deltaSeconds   time-slot length (Δt)
     */
    public void tick(int hostId, double utilizationPct, double deltaSeconds) {
        double util = Math.max(0.0, Math.min(100.0, utilizationPct)) / 100.0;
        double drawW = idlePowerW[hostId] + (maxPowerW[hostId] - idlePowerW[hostId]) * util;
        remainingJ[hostId] = Math.max(0.0, remainingJ[hostId] - drawW * deltaSeconds);
    }

    /** Remaining battery as a percentage of capacity (0-100) — this is E_i^t in the state vector. */
    public double getRemainingPct(int hostId) {
        return 100.0 * remainingJ[hostId] / capacityJ[hostId];
    }

    /** True once a host's battery has reached zero (Constraint C5-style hard failure). */
    public boolean isDead(int hostId) {
        return remainingJ[hostId] <= 0.0;
    }

    public int getNumberOfHost() {
        return numberOfHost;
    }
}
