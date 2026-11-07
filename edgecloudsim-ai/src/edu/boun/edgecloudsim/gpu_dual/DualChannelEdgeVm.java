/*
 * Title:        EdgeCloudSim - DualChannelEdgeVm  (Phase 2)
 *
 * Description:
 * Edge VM for Phase 2 dual-channel simulation.
 * Separates CPU MIPS (ARM Cortex-A57) from GPU MFLOPS (TX2 GPU).
 *
 * Key design difference vs Phase 1 GpuVm:
 *   Phase 1 GpuVm  — VM.mips = gpuMips = 135,000  (single reference)
 *   Phase 2 here   — VM.mips = gpuMips = 135,000   (same, for getAvgUtilization compatibility)
 *                    cpuMips = 3,000                (new field, separate from VM.mips)
 *
 * VM.mips must stay at GPU MFLOPS so CloudSim's:
 *   processingTime = effectiveMI / VM.mips = effectiveTime ✓
 * ... and getAvgUtilization() still reflects GPU load correctly.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.gpu_dual;

import org.cloudbus.cloudsim.CloudletScheduler;

import edu.boun.edgecloudsim.edge_server.EdgeVM;

/**
 * Dual-channel capable Edge VM (Phase 2).
 *
 * <p>Extends {@link EdgeVM} with:
 * <ul>
 *   <li>{@code cpuMips}   — ARM CPU throughput, passed to {@link DualChannelScheduler}</li>
 *   <li>{@code gpuMips}   — GPU MFLOPS; equals VM.mips for CloudSim compatibility</li>
 *   <li>{@code vramMB}    — VRAM for admission control (same as Phase 1)</li>
 * </ul>
 * </p>
 */
public class DualChannelEdgeVm extends EdgeVM {

    /** ARM Cortex-A57 CPU throughput in MIPS (e.g. 3,000). */
    private final long cpuMips;

    /** GPU throughput in MFLOPS (e.g. 135,000 for TX2). Also = VM.mips. */
    private final long gpuMips;

    /** Total VRAM / unified memory in MB (e.g. 8,192 for TX2). */
    private final long vramMB;

    /** Currently allocated VRAM in MB. */
    private long usedVramMB = 0;

    /**
     * Device power budget in Watts (e.g. 15.0 for TX2) — used by
     * edu.boun.edgecloudsim.drl.EdgeEnergyManager for the DRL battery model.
     * Not used anywhere in tutorial3_ai_flop's own orchestrator.
     */
    private final double powerW;

    /**
     * @param id       unique VM id
     * @param userId   broker id
     * @param mips     VM.mips — must equal gpuMips for CloudSim formula correctness
     * @param cores    number of CPU cores
     * @param ram      RAM in MB
     * @param bw       bandwidth in Mbps
     * @param size     storage in MB
     * @param vmm      hypervisor type string
     * @param sched    {@link DualChannelScheduler} pre-configured with cpuMips
     * @param cpuMips  ARM CPU MIPS (e.g. 3000)
     * @param gpuMips  GPU MFLOPS reference — same as mips (e.g. 135000)
     * @param vramMB   total VRAM in MB (e.g. 8192)
     * @param powerW   device power budget in Watts (e.g. 15.0 for TX2)
     */
    public DualChannelEdgeVm(int id, int userId, double mips, int cores,
                              int ram, long bw, long size, String vmm,
                              CloudletScheduler sched,
                              long cpuMips, long gpuMips, long vramMB, double powerW) {
        super(id, userId, mips, cores, ram, bw, size, vmm, sched);
        this.cpuMips = cpuMips;
        this.gpuMips = gpuMips;
        this.vramMB  = vramMB;
        this.powerW  = powerW;
    }

    public long getCpuMips()    { return cpuMips; }
    public long getGpuMips()    { return gpuMips; }
    public long getTotalVramMB(){ return vramMB; }
    public long getUsedVramMB() { return usedVramMB; }
    public long getFreeVramMB() { return Math.max(0, vramMB - usedVramMB); }
    public double getPowerW()   { return powerW; }

    public synchronized boolean canFitVram(long mb) {
        return mb <= 0 || getFreeVramMB() >= mb;
    }

    public synchronized void allocateVram(long mb) {
        if (mb > 0) {
			usedVramMB = Math.min(vramMB, usedVramMB + mb);
		}
    }

    public synchronized void releaseVram(long mb) {
        if (mb > 0) {
			usedVramMB = Math.max(0, usedVramMB - mb);
		}
    }
}
