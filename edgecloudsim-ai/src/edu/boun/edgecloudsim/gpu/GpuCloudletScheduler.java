/*
 * Title:        EdgeCloudSim - GpuCloudletScheduler
 *
 * Description:
 * SpaceShared scheduler for GPU-capable edge VMs.
 *
 * Physical accuracy: a GPU processes one batch at a time (space-shared),
 * not interleaved like a CPU time-sharing scheduler. This class enforces
 * that behaviour by extending CloudletSchedulerSpaceShared.
 *
 * cloudletLength (MFLOP) / VM.mips (MFLOPS) = processingTime (s)  ← CloudSim formula
 *
 * For the current single-channel approach:
 *   cloudletLength = AITaskProperty.computeGpuMFLOP() (GPU MFLOP with overheadFactor)
 *   VM.mips        = gpuMips = 135,000 MFLOPS (Jetson TX2)
 *   processingTime = 10178 / 135000 = 75 ms  for ResNet-50  ✓
 *
 * Phase-2 note (dual-channel — NOT enabled yet):
 *   To enable max(cpuTime, gpuTime) scheduling, change AITaskProperty to set
 *   cloudletLength = computeCpuMI() (CPU MI only), and add dual-channel logic
 *   to cloudletSubmit() that rewrites cloudletLength = max(cpuTime, gpuTime) × vmMips.
 *   This requires also upgrading the cloud scheduler (DefaultCloudServerManager).
 *   The Task.tensors field is already wired up and ready for this extension.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.gpu;

import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;

/**
 * GPU-aware SpaceShared Cloudlet Scheduler.
 *
 * <p>Extends {@link CloudletSchedulerSpaceShared} — GPUs run one batch at a time
 * (no time-sharing), so SpaceShared is the physically correct scheduling model.</p>
 *
 * <p>Carries a {@code gpuMips} field that matches the {@link GpuVm}'s GPU throughput.
 * Currently the scheduler inherits all behaviour from the parent without overriding
 * {@code cloudletSubmit()} — cloudletLength already encodes the correct GPU execution
 * time via {@code AITaskProperty.computeGpuMFLOP()} and VM.mips = gpuMips.</p>
 *
 * <p><b>Why SpaceShared over TimeShared?</b><br>
 * {@code CloudletSchedulerTimeShared} (used by DefaultEdgeServerManager) shares GPU
 * cycles among concurrent tasks, which is not physically meaningful for GPU inference.
 * SpaceShared queues tasks and runs them one after another — matching real GPU behaviour
 * where concurrent inference requests share GPU time via a FIFO hardware queue.</p>
 *
 * <p><b>Integration:</b> Created by {@link GpuEdgeServerManager} and passed to
 * each {@link GpuVm}. No changes to any other CloudSim class are required.</p>
 */
public class GpuCloudletScheduler extends CloudletSchedulerSpaceShared {

    /**
     * GPU throughput in MFLOPS — matches the owning {@link GpuVm}'s gpuMips.
     * For Jetson TX2: 135,000 MFLOPS.
     * Stored here for scheduler-side queries (e.g., Phase-2 dual-channel logic).
     */
    private final long gpuMips;

    /**
     * Creates a GpuCloudletScheduler with the given GPU MFLOPS rating.
     *
     * @param gpuMips GPU throughput in MFLOPS (e.g. 135000 for Jetson TX2)
     */
    public GpuCloudletScheduler(long gpuMips) {
        super();
        this.gpuMips = gpuMips;
    }

    /**
     * Returns the GPU throughput in MFLOPS.
     * @return GPU MFLOPS (e.g. 135000 for Jetson TX2)
     */
    public long getGpuMips() {
        return gpuMips;
    }
}
