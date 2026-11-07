/*
 * Title:        EdgeCloudSim - DRLEdgeOrchestrator
 *
 * Description:
 * Replaces tutorial3.SampleEdgeOrchestrator entirely for the SAC-driven
 * scenario. Implements Algorithm 2 (Workload Execution and Strict RAM
 * Constraint Verification) from PROTOCOL.md / introduction.md: every
 * DELTA_T_SECONDS it exchanges one state/action round-trip with the
 * Python SAC controller over a DRLSocketBridge, then uses the cached
 * N x (N+1) action matrix (N Edge Hosts + Cloud) to route each
 * individually arriving task by sampling from its origin host's row
 * as a categorical distribution.
 *
 * Built on the tutorial3_ai_flop track (gpu_dual / DualChannelScheduler --
 * GPU/CPU workload set directly in applications.xml, VM.mips = GPU MFLOPS
 * reference) rather than gpu_ai/AITask/AIWorkloadCalculator: VRAM admission
 * control is delegated directly to DualChannelEdgeVm's own
 * canFitVram()/allocateVram()/releaseVram() (already built into that
 * track), not reimplemented here. GPU load is read via
 * getTotalUtilizationOfCpu() -- which reflects GPU load in this track
 * because VM.mips is pinned to GPU MFLOPS, see DualChannelEdgeVm's own
 * class doc. CPU load has no native CloudSim tracking in this design (CPU
 * MI is only used once at submission inside DualChannelScheduler to
 * compute max(cpuTime, gpuTime), never tracked as an ongoing occupancy
 * percentage), so it is estimated here from a small per-VM backlog of
 * currently-admitted cloudletLength (CPU MI), see computeCpuUsedPct().
 *
 * One instance of this class lives for exactly one SAC episode (T_max
 * time-slots): it is constructed fresh per episode by DRLScenarioFactory,
 * alongside a fresh SimManager/CloudSim run, and reuses the same
 * long-lived DRLSocketBridge across all episodes.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.drl;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;

import edu.boun.edgecloudsim.cloud_server.CloudVM;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.CpuUtilizationModel_Custom;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeHost;
import edu.boun.edgecloudsim.edge_server.EdgeVM;
import edu.boun.edgecloudsim.gpu_dual.DualChannelEdgeVm;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;

public class DRLEdgeOrchestrator extends EdgeOrchestrator {

    /** Time-slot length (seconds of simulated time) — see PROTOCOL.md section 6. */
    private static final double DELTA_T_SECONDS = 5.0;

    /** Arbitrary tag, offset to avoid collisions with other SimEntity event tags in the project. */
    private static final int TIME_SLOT_TICK = 300001;

    private final DRLSocketBridge bridge;
    private final int episode;

    private int numberOfHost;
    private EdgeEnergyManager energyManager;
    private Map<Integer, Integer> wlanToHostId;

    private double[][] currentActionMatrix;
    private int timeSlot;
    private int tMaxSlots;
    private boolean shutdownRequested = false;

    private final List<Double> completedLatenciesMs = new java.util.ArrayList<>();
    private int dropCountThisSlot = 0;
    private int cloudRoutedCountThisSlot = 0;
    private int edgeRoutedCountThisSlot = 0;

    /** Bridges getDeviceToOffload() -> getVmToOffload(): cloudletId -> chosen column j. */
    private final Map<Integer, Integer> pendingHostChoice = new HashMap<>();

    /**
     * Per-VM wall-clock seconds of CPU/GPU-channel work admitted during the
     * CURRENT time-slot only (reset every tick, see resetSlotAccumulators()) --
     * vmId -> accumulated seconds. Used to estimate C_i/G_i (see
     * computeCpuUsedPct/computeGpuUsedPct) as a throughput measure.
     *
     * <p>Earlier version sampled {@code CloudletScheduler.getTotalUtilizationOfCpu()}
     * (or a never-reset outstanding-backlog balance) at the tick instant. Both are
     * point-in-time snapshots, but each task here only executes for ~tens of ms
     * (see DualChannelScheduler's effectiveTime) against a DELTA_T_SECONDS=5s tick
     * -- so a snapshot almost always lands on an idle instant regardless of actual
     * load, making Q_i/E_i nearly identical across hosts no matter how skewed
     * routing is (confirmed by direct socket inspection: Jain's Fairness stayed
     * exactly 1.000000 even with Q_i ranging 0-11 across hosts at 1500 devices).
     * Accumulating admitted work-seconds over the whole slot instead captures
     * actual throughput, not a lucky/unlucky sample.</p>
     */
    private final Map<Integer, Double> vmCpuBusySecThisSlot = new HashMap<>();
    private final Map<Integer, Double> vmGpuBusySecThisSlot = new HashMap<>();

    public DRLEdgeOrchestrator(String policy, String simScenario,
                                DRLSocketBridge bridge, int episode) {
        super(policy, simScenario);
        this.bridge = bridge;
        this.episode = episode;
    }

    // =======================================================================
    // Setup
    // =======================================================================

    @Override
    public void initialize() {
        // NOTE: called from inside SimManager's constructor, BEFORE
        // SimManager.instance is assigned and BEFORE edgeServerManager is
        // even constructed (see SimManager.java: getEdgeOrchestrator() +
        // initialize() run before edgeServerManager is built, and its
        // datacenters/VMs only exist after startSimulation() calls
        // edgeServerManager.startDatacenters()/createVmList() -- well after
        // this method returns). So only config-driven setup belongs here;
        // anything touching SimManager.getInstance() or the edge server
        // topology (including EdgeEnergyManager, which now needs each
        // host's <power_w> read off its DualChannelEdgeVm) must wait for
        // startEntity().
        numberOfHost = SimSettings.getInstance().getNumOfEdgeHosts();
        tMaxSlots = (int) Math.floor(usableSeconds() / DELTA_T_SECONDS);
    }

    private double usableSeconds() {
        SimSettings ss = SimSettings.getInstance();
        return ss.getSimulationTime() - ss.getWarmUpPeriod();
    }

    /**
     * Walks datacenters/hosts in the same nested order DefaultEdgeServerManager
     * assigns its global host id counter, so the index built here matches the
     * hostId expected by EdgeServerManager.getVmList(hostId). Must be called
     * from startEntity() or later -- see the note on initialize().
     */
    private Map<Integer, Integer> buildWlanToHostMap() {
        Map<Integer, Integer> map = new HashMap<>();
        int hostId = 0;
        for (org.cloudbus.cloudsim.Datacenter dc : SimManager.getInstance()
                .getEdgeServerManager().getDatacenterList()) {
            for (Host h : dc.getHostList()) {
                int wlanId = ((EdgeHost) h).getLocation().getServingWlanId();
                map.put(wlanId, hostId);
                hostId++;
            }
        }
        return map;
    }

    /**
     * Reads each host's power budget (Watts) from its first VM's <power_w>
     * (DualChannelEdgeVm.getPowerW(), see edge_devices.xml). All VMs on a
     * host share the same physical device in this simulation model, same
     * assumption already made for cpu_mips/gpu_mips/vram_mb. Must be called
     * from startEntity() or later -- see the note on initialize().
     */
    private double[] buildPowerBudgetPerHost() {
        double[] powerW = new double[numberOfHost];
        for (int i = 0; i < numberOfHost; i++) {
            List<EdgeVM> vms = getVmList(i);
            powerW[i] = (!vms.isEmpty() && vms.get(0) instanceof DualChannelEdgeVm)
                    ? ((DualChannelEdgeVm) vms.get(0)).getPowerW()
                    : 15.0; // fallback matches DualChannelEdgeServerManager.DEFAULT_POWER_W
        }
        return powerW;
    }

    // =======================================================================
    // SimEntity lifecycle — drives the per-time-slot socket round-trip
    // =======================================================================

    @Override
    public void startEntity() {
        wlanToHostId = buildWlanToHostMap();
        energyManager = new EdgeEnergyManager(buildPowerBudgetPerHost(), usableSeconds());

        try {
            double[] s1 = buildStateVector();
            bridge.sendEpisodeStart(episode, numberOfHost, s1.length, s1);
        } catch (IOException e) {
            throw new RuntimeException("DRL socket failure at episode start", e);
        }

        double[][] a1 = tryReceiveAction();
        if (a1 == null) {
            // Shutdown requested before any action was ever received for this
            // episode — fall back to a safe "everything stays local" matrix
            // instead of leaving currentActionMatrix null.
            shutdownRequested = true;
            currentActionMatrix = buildLocalOnlyMatrix();
            return; // no ticks scheduled; episode runs out using the fallback only
        }
        currentActionMatrix = a1;
        schedule(getId(), usableSeconds() > 0 ? DELTA_T_SECONDS : 0, TIME_SLOT_TICK);
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == TIME_SLOT_TICK) {
            onTimeSlotTick();
        }
    }

    @Override
    public void shutdownEntity() { /* socket lifecycle owned by DRLMainApp, spans all episodes */ }

    /** True once Python has sent "shutdown" instead of an action (see PROTOCOL.md 4.4). */
    public boolean isShutdownRequested() {
        return shutdownRequested;
    }

    private void onTimeSlotTick() {
        timeSlot++;
        boolean done = timeSlot >= tMaxSlots;

        for (int i = 0; i < numberOfHost; i++) {
            double combinedUtilPct = 0.5 * computeCpuUsedPct(i) + 0.5 * computeGpuUsedPct(i);
            energyManager.tick(i, combinedUtilPct, DELTA_T_SECONDS);
        }

        double[] nextState = buildStateVector();
        double[] latencies = completedLatenciesMs.stream().mapToDouble(Double::doubleValue).toArray();

        try {
            bridge.sendTransition(episode, timeSlot, nextState, latencies, dropCountThisSlot, done,
                    cloudRoutedCountThisSlot, edgeRoutedCountThisSlot);
        } catch (IOException e) {
            throw new RuntimeException("DRL socket failure sending transition", e);
        }
        completedLatenciesMs.clear();
        dropCountThisSlot = 0;
        cloudRoutedCountThisSlot = 0;
        edgeRoutedCountThisSlot = 0;
        vmCpuBusySecThisSlot.clear();
        vmGpuBusySecThisSlot.clear();

        if (!done) {
            double[][] next = tryReceiveAction();
            if (next == null) {
                // Early stop mid-episode: freeze routing to "local only" and stop
                // ticking. CloudSim keeps running to its natural simulation_time
                // end using that safe fallback; DRLMainApp checks
                // isShutdownRequested() after startSimulation() returns and does
                // not start another episode.
                shutdownRequested = true;
                currentActionMatrix = buildLocalOnlyMatrix();
                return;
            }
            currentActionMatrix = next;
            schedule(getId(), DELTA_T_SECONDS, TIME_SLOT_TICK);
        }
        // if done: no reschedule, no action wait — episode ends here; DRLMainApp
        // starts the next episode (fresh SimManager + fresh DRLEdgeOrchestrator)
        // over the same DRLSocketBridge.
    }

    /** Returns the next action matrix, or null if Python sent "shutdown" (see PROTOCOL.md 4.4). */
    private double[][] tryReceiveAction() {
        try {
            return bridge.receiveActionOrNullOnShutdown();
        } catch (IOException e) {
            throw new RuntimeException("DRL socket failure receiving action", e);
        }
    }

    /** Every host routes 100% of its own traffic to itself — safe degenerate fallback. */
    private double[][] buildLocalOnlyMatrix() {
        double[][] m = new double[numberOfHost][numberOfHost + 1];
        for (int i = 0; i < numberOfHost; i++) {
            m[i][i] = 1.0;
        }
        return m;
    }

    // =======================================================================
    // State vector (see PROTOCOL.md section 2)
    // =======================================================================

    private double[] buildStateVector() {
        double[] state = new double[numberOfHost * 5 + 1];
        for (int i = 0; i < numberOfHost; i++) {
            int base = i * 5;
            state[base]     = computeQueueLength(i);
            state[base + 1] = 100.0 - computeCpuUsedPct(i);
            state[base + 2] = 100.0 - computeGpuUsedPct(i);
            state[base + 3] = computeFreeRamGB(i);
            state[base + 4] = energyManager.getRemainingPct(i);
        }
        state[state.length - 1] = tMaxSlots > 0 ? (timeSlot / (double) tMaxSlots) : 0.0;
        return state;
    }

    /** Q_i: tasks currently queued or executing at this host, read straight from CloudSim's own scheduler state. */
    private int computeQueueLength(int hostId) {
        int total = 0;
        for (EdgeVM vm : getVmList(hostId)) {
            total += vm.getCloudletScheduler().runningCloudlets();
            total += vm.getCloudletScheduler().getCloudletWaitingList().size();
        }
        return total;
    }

    /**
     * G_i used%: fraction of this slot's Δt that each VM's GPU channel was busy,
     * i.e. sum(admitted gpuTime this slot) / Δt, averaged across the host's VMs.
     * See vmGpuBusySecThisSlot's class doc for why this replaces an instantaneous
     * CloudletScheduler sample.
     */
    private double computeGpuUsedPct(int hostId) {
        return averageBusyPct(hostId, vmGpuBusySecThisSlot);
    }

    /**
     * C_i used%: fraction of this slot's Δt that each VM's CPU channel was busy,
     * i.e. sum(admitted cpuTime this slot) / Δt, averaged across the host's VMs.
     * CloudSim never tracks CPU-channel occupancy natively in this dual-channel
     * design (CPU MI is only used once at submission inside DualChannelScheduler
     * to compute max(cpuTime, gpuTime)), so this is self-tracked (see
     * vmCpuBusySecThisSlot).
     */
    private double computeCpuUsedPct(int hostId) {
        return averageBusyPct(hostId, vmCpuBusySecThisSlot);
    }

    private double averageBusyPct(int hostId, Map<Integer, Double> busySecByVm) {
        List<EdgeVM> vms = getVmList(hostId);
        if (vms.isEmpty()) return 0.0;
        double totalPct = 0.0;
        for (EdgeVM vm : vms) {
            double busySec = busySecByVm.getOrDefault(vm.getId(), 0.0);
            totalPct += Math.min(100.0, (busySec / DELTA_T_SECONDS) * 100.0);
        }
        return totalPct / vms.size();
    }

    /** R_i: total free VRAM across this host's VMs, in GB — delegates to DualChannelEdgeVm directly. */
    private double computeFreeRamGB(int hostId) {
        double freeMb = 0.0;
        for (EdgeVM vm : getVmList(hostId)) {
            if (vm instanceof DualChannelEdgeVm) {
                freeMb += ((DualChannelEdgeVm) vm).getFreeVramMB();
            }
        }
        return freeMb / 1024.0;
    }

    private List<EdgeVM> getVmList(int hostId) {
        return SimManager.getInstance().getEdgeServerManager().getVmList(hostId);
    }

    // =======================================================================
    // Algorithm 2 — per-task routing decisions
    // =======================================================================

    @Override
    public int getDeviceToOffload(Task task) {
        int homeHostId = resolveHomeHostId(task);
        int j = sampleDestination(currentActionMatrix[homeHostId]);
        pendingHostChoice.put(task.getCloudletId(), j);

        if (j == numberOfHost) {
            cloudRoutedCountThisSlot++;
            return SimSettings.CLOUD_DATACENTER_ID;
        }
        edgeRoutedCountThisSlot++;
        return SimSettings.GENERIC_EDGE_DEVICE_ID;
    }

    private int resolveHomeHostId(Task task) {
        Integer homeHostId = wlanToHostId.get(task.getSubmittedLocation().getServingWlanId());
        if (homeHostId == null) {
            SimLogger.printLine("DRLEdgeOrchestrator: no host found for wlan_id="
                    + task.getSubmittedLocation().getServingWlanId() + "; defaulting to host 0");
            return 0;
        }
        return homeHostId;
    }

    /** Samples a column index from a row of the action matrix, treated as a categorical distribution. */
    private int sampleDestination(double[] rowProbabilities) {
        double r = SimUtils.getRandomDoubleNumber(0, 1);
        double cumulative = 0.0;
        for (int j = 0; j < rowProbabilities.length; j++) {
            cumulative += rowProbabilities[j];
            if (r <= cumulative) {
                return j;
            }
        }
        return rowProbabilities.length - 1; // floating-point safety net -> Cloud column
    }

    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        Integer j = pendingHostChoice.remove(task.getCloudletId());
        if (j == null) {
            SimLogger.printLine("DRLEdgeOrchestrator: no cached routing decision for cloudlet "
                    + task.getCloudletId());
            return null;
        }

        if (deviceId == SimSettings.CLOUD_DATACENTER_ID) {
            return selectLeastLoadedCloudVm(task);
        }

        int homeHostId = resolveHomeHostId(task);
        boolean isRemote = (j != homeHostId);

        // Energy constraint (agreed extension of Constraint C5): a dead host cannot
        // compute, whether serving its own local task or a remote one.
        if (energyManager.isDead(j)) {
            dropCountThisSlot++;
            return null;
        }

        long vramMB = task.getVramRequired();
        // RAM constraint (Algorithm 2, Constraint C5): applies to the remote path
        // only, matching the pseudocode exactly (local admission is unconditional there).
        EdgeVM vm = isRemote
                ? selectVmWithVramFit(task, j, vramMB)
                : selectLeastLoadedVmOnHost(task, j);

        if (vm == null) {
            if (isRemote) {
                dropCountThisSlot++; // no VM on host j had enough free VRAM
            }
            return null;
        }

        admit(vm, task, vramMB);
        return vm;
    }

    /**
     * Accumulates this task's CPU/GPU busy-seconds into the current slot's
     * throughput tally (see vmCpuBusySecThisSlot's class doc) and reserves VRAM
     * on the VM the task was actually admitted to.
     */
    private void admit(EdgeVM vm, Task task, long vramMB) {
        if (vm instanceof DualChannelEdgeVm) {
            DualChannelEdgeVm dcVm = (DualChannelEdgeVm) vm;
            // Snapshot BEFORE DualChannelScheduler.cloudletSubmit() overwrites
            // cloudletLength with the GPU-MFLOPS-scaled effectiveMI -- mirrors the
            // same cpuTime/gpuTime/effectiveTime formula computed there, just kept
            // separate per channel instead of collapsed into max().
            long cpuMi = task.getCloudletLength();
            long tensors = task.getTensors();
            long cpuMips = dcVm.getCpuMips();
            long gpuMips = dcVm.getGpuMips();
            double cpuTime = cpuMips > 0 ? cpuMi / (double) cpuMips : 0.0;
            double gpuTime = (tensors > 0 && gpuMips > 0) ? tensors / (double) gpuMips : 0.0;
            vmCpuBusySecThisSlot.merge(vm.getId(), cpuTime, Double::sum);
            vmGpuBusySecThisSlot.merge(vm.getId(), gpuTime, Double::sum);
            dcVm.allocateVram(vramMB);
        }
    }

    private EdgeVM selectLeastLoadedVmOnHost(Task task, int hostId) {
        EdgeVM selected = null;
        double bestAvailable = -1.0;
        for (EdgeVM vm : getVmList(hostId)) {
            double required = ((CpuUtilizationModel_Custom) task.getUtilizationModelCpu())
                    .predictUtilization(vm.getVmType());
            double available = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
            if (required <= available && available > bestAvailable) {
                selected = vm;
                bestAvailable = available;
            }
        }
        return selected;
    }

    /** Same least-loaded selection, restricted to VMs with enough free VRAM for this task. */
    private EdgeVM selectVmWithVramFit(Task task, int hostId, long vramMB) {
        EdgeVM selected = null;
        double bestAvailable = -1.0;
        for (EdgeVM vm : getVmList(hostId)) {
            if (!(vm instanceof DualChannelEdgeVm) || !((DualChannelEdgeVm) vm).canFitVram(vramMB)) {
                continue;
            }
            double required = ((CpuUtilizationModel_Custom) task.getUtilizationModelCpu())
                    .predictUtilization(vm.getVmType());
            double available = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
            if (required <= available && available > bestAvailable) {
                selected = vm;
                bestAvailable = available;
            }
        }
        return selected;
    }

    private Vm selectLeastLoadedCloudVm(Task task) {
        Vm selected = null;
        double bestAvailable = 0.0;
        List<Host> list = SimManager.getInstance().getCloudServerManager().getDatacenter().getHostList();
        for (int h = 0; h < list.size(); h++) {
            List<CloudVM> vms = SimManager.getInstance().getCloudServerManager().getVmList(h);
            for (CloudVM vm : vms) {
                double required = ((CpuUtilizationModel_Custom) task.getUtilizationModelCpu())
                        .predictUtilization(vm.getVmType());
                double available = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
                if (required <= available && available > bestAvailable) {
                    selected = vm;
                    bestAvailable = available;
                }
            }
        }
        return selected;
    }

    // =======================================================================
    // Completion hook — called by DRLMobileDeviceManager.processCloudletReturn()
    // =======================================================================

    /**
     * @param hostId        global Edge Host index the task executed on, ignored if !executedOnEdge
     * @param latencyMs     upload+queue+exec latency (see DRLMobileDeviceManager note on download delay)
     * @param executedOnEdge false if the task ran on Cloud (no per-VM resource to release)
     */
    public void onTaskCompleted(Task task, int hostId, double latencyMs, boolean executedOnEdge) {
        completedLatenciesMs.add(latencyMs);
        // VRAM release is handled by DRLMobileDeviceManager (it resolves the exact
        // VM instance). Busy-seconds need no completion-time bookkeeping: they are
        // tallied once at admit() and reset every slot (see vmCpuBusySecThisSlot's
        // class doc), not maintained as a running outstanding balance.
    }
}
