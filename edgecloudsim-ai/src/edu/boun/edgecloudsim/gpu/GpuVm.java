/*
 * Title:        EdgeCloudSim - GpuVm
 *
 * Description:
 * Extends EdgeVM with GPU-specific resources: GPU MFLOPS and VRAM.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.gpu;

import org.cloudbus.cloudsim.CloudletScheduler;

import edu.boun.edgecloudsim.edge_server.EdgeVM;

/**
 * GPU-capable VM for edge computing simulation.
 *
 * <p>Extends {@link EdgeVM} with two additional resource dimensions:
 * <ul>
 *   <li><b>gpuMips</b>  — GPU throughput in MFLOPS (Million FLOPs/s).
 *       For Jetson TX2 FP32: 135,000 MFLOPS. Injected into
 *       {@link GpuCloudletScheduler} at construction time.</li>
 *   <li><b>totalVramMB</b> — Total GPU/unified memory in MB.
 *       For Jetson TX2: 8,192 MB (8 GB unified LPDDR4).
 *       Used for admission control: reject tasks that exceed available VRAM.</li>
 * </ul>
 * </p>
 *
 * <p><b>VRAM tracking:</b> {@code canFitVram()}, {@code allocateVram()},
 * and {@code releaseVram()} enable per-task VRAM admission control.
 * VRAM is allocated when a task is submitted to the edge VM and released
 * when the task completes (CLOUDLET_RETURN event in MobileDeviceManager).</p>
 *
 * <p><b>Bandwidth sharing:</b> inherits EdgeVM's proportional WLAN bandwidth.</p>
 */
public class GpuVm extends EdgeVM {

    /** GPU throughput in MFLOPS (Million FLOPs/s). */
    private final long gpuMips;

    /** Total GPU / unified memory available in MB. */
    private final long totalVramMB;

    /** Currently allocated VRAM in MB (updated by allocate/releaseVram). */
    private long usedVramMB = 0;

    /**
     * Constructs a GpuVm with full resource specification.
     *
     * @param id                unique VM id
     * @param userId            broker/user id
     * @param mips              CPU MIPS (VM.mips — also serves as GPU MFLOPS reference
     *                          in CloudSim's processingTime = length/mips formula)
     * @param numberOfPes       number of CPU cores
     * @param ram               RAM in MB
     * @param bw                bandwidth in Mbps
     * @param size              storage in MB
     * @param vmm               hypervisor type string
     * @param cloudletScheduler {@link GpuCloudletScheduler} with dual-channel logic
     * @param gpuMips           GPU throughput in MFLOPS (e.g. 135000 for TX2)
     * @param totalVramMB       total VRAM / unified memory in MB (e.g. 8192 for TX2)
     */
    public GpuVm(int id, int userId, double mips, int numberOfPes,
                 int ram, long bw, long size, String vmm,
                 CloudletScheduler cloudletScheduler,
                 long gpuMips, long totalVramMB) {
        super(id, userId, mips, numberOfPes, ram, bw, size, vmm, cloudletScheduler);
        this.gpuMips      = gpuMips;
        this.totalVramMB  = totalVramMB;
    }

    // -----------------------------------------------------------------------
    // GPU resource accessors
    // -----------------------------------------------------------------------

    /** @return GPU throughput in MFLOPS */
    public long getGpuMips()      { return gpuMips; }

    /** @return total VRAM / unified memory in MB */
    public long getTotalVramMB()  { return totalVramMB; }

    /** @return currently allocated VRAM in MB */
    public long getUsedVramMB()   { return usedVramMB; }

    /** @return free VRAM in MB (never negative) */
    public long getFreeVramMB()   { return Math.max(0, totalVramMB - usedVramMB); }

    // -----------------------------------------------------------------------
    // VRAM admission control
    // -----------------------------------------------------------------------

    /**
     * Returns {@code true} if {@code mb} MB of VRAM is currently available.
     * @param mb VRAM needed in MB
     */
    public synchronized boolean canFitVram(long mb) {
        return mb <= 0 || getFreeVramMB() >= mb;
    }

    /**
     * Allocates {@code mb} MB of VRAM for a task starting execution.
     * Clamps to totalVramMB to prevent overflow from concurrent updates.
     * @param mb VRAM to allocate in MB
     */
    public synchronized void allocateVram(long mb) {
        if (mb > 0) {
            usedVramMB = Math.min(totalVramMB, usedVramMB + mb);
        }
    }

    /**
     * Releases {@code mb} MB of VRAM when a task finishes execution.
     * Clamps to zero to prevent underflow.
     * @param mb VRAM to release in MB
     */
    public synchronized void releaseVram(long mb) {
        if (mb > 0) {
            usedVramMB = Math.max(0, usedVramMB - mb);
        }
    }

    // -----------------------------------------------------------------------
    // Convenience
    // -----------------------------------------------------------------------

    @Override
    public String toString() {
        return String.format("GpuVm[id=%d, gpuMips=%d, vram=%d/%d MB, util=%.1f%%]",
            getId(), gpuMips, usedVramMB, totalVramMB,
            getCloudletScheduler().getTotalUtilizationOfCpu(0) * 100);
    }
}
