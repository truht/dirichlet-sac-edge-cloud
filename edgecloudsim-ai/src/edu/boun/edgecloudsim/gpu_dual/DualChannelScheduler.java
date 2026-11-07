/*
 * Title:        EdgeCloudSim - DualChannelScheduler  (Phase 2)
 *
 * Description:
 * Implements true dual-channel GPU scheduling:
 *   processingTime = max( cpuTime, gpuTime )
 *
 * This is the core of Phase 2.  It uses setCloudletLength() — confirmed
 * present in cloudsim-7.0.0-alpha.jar — to rewrite the cloudlet's length
 * just before handing it to the SpaceShared parent, so CloudSim's internal
 *   processingTime = length / vmMips
 * yields the correct wall-clock result.
 *
 * Channel separation (Phase 2):
 *   cloudletLength (MI)    = CPU preprocessing  → set by AITaskPropertyDual.computeCpuMI()
 *   tensors        (MFLOP) = GPU inference       → set by AILoadGeneratorDual / AITask Phase-2 constructor
 *   VM.mips                = GPU MFLOPS reference (kept at GPU MFLOPS so getAvgUtilization() still works)
 *   cpuMips                = actual CPU MIPS     (passed in constructor — separate from VM.mips)
 *
 * Formula:
 *   cpuTime       = cloudletLength / cpuMips           e.g. 24 MI   / 3,000     =  8 ms  (TX2 ARM)
 *   gpuTime       = tensors        / vmMips             e.g. 10,178  / 135,000   = 75 ms  (TX2 GPU)
 *   effectiveTime = max(cpuTime, gpuTime)                                        = 75 ms
 *   effectiveMI   = effectiveTime  × vmMips             e.g. 75ms    × 135,000   = 10,125
 *   → setCloudletLength(effectiveMI)
 *   → super.cloudletSubmit()
 *   → CloudSim: processingTime = 10,125 / 135,000 = 75 ms ✓
 *
 * Cloud T4 example:
 *   cpuTime =  24 MI   / 100,000   =  0.24 ms  (Intel Xeon preprocessing)
 *   gpuTime = 10,178   / 3,500,000 =  2.9  ms  (T4 GPU inference)
 *   effectiveTime = 2.9 ms ✓
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.gpu_dual;

import java.util.List;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;

import edu.boun.edgecloudsim.edge_client.Task;

/**
 * Dual-channel SpaceShared Cloudlet Scheduler (Phase 2).
 *
 * <p><b>vs Phase 1 ({@code GpuCloudletScheduler}):</b><br>
 * Phase 1 sets {@code cloudletLength = GPU MFLOP} in {@code AITaskProperty}
 * and does not override {@code cloudletSubmit()} — single channel only.<br>
 * Phase 2 keeps {@code cloudletLength = CPU MI}, stores GPU MFLOP in
 * {@code task.tensors}, and this class computes {@code max(cpu, gpu)} here
 * before CloudSim sees the task.</p>
 *
 * <p><b>Works for both edge and cloud</b> — parametrized by {@code cpuMips}:
 * <ul>
 *   <li>Edge (TX2 ARM Cortex-A57): {@code cpuMips = 3,000}</li>
 *   <li>Cloud (Intel Xeon):        {@code cpuMips = 100,000}</li>
 * </ul>
 * {@code vmMips} (GPU MFLOPS reference) is read from {@code getCurrentMipsShare()}
 * at submit time — works for both TX2 (135,000) and T4 (3,500,000).</p>
 */
public class DualChannelScheduler extends CloudletSchedulerSpaceShared {

    /** CPU MIPS of the host processor (ARM Cortex-A57 = 3,000; Intel Xeon = 100,000). */
    private final long cpuMips;

    /**
     * Creates a DualChannelScheduler with the given CPU MIPS.
     *
     * @param cpuMips CPU throughput in MIPS (e.g. 3000 for TX2, 100000 for cloud Xeon)
     */
    public DualChannelScheduler(long cpuMips) {
        super();
        this.cpuMips = cpuMips;
    }

    /**
     * Intercepts task submission to apply dual-channel max(cpuTime, gpuTime) model.
     *
     * <p>Rewrites {@code cloudletLength} so CloudSim computes the correct
     * wall-clock time via its standard {@code processingTime = length / vmMips}.</p>
     *
     * <p>Falls through to {@code super.cloudletSubmit()} unchanged if:
     * <ul>
     *   <li>task is not a {@link Task} instance</li>
     *   <li>{@code task.getTensors() == 0} (non-GPU task)</li>
     *   <li>{@code getCurrentMipsShare()} is null/empty (VM not yet attached)</li>
     * </ul>
     * </p>
     */
    @Override
    public double cloudletSubmit(Cloudlet cl, double fileTransferTime) {

        if (cl instanceof Task) {
            Task task = (Task) cl;
            long tensors = task.getTensors(); // GPU MFLOP (set by AILoadGeneratorDual / Phase-2 AITask)

            if (tensors > 0 && cpuMips > 0) {
                List<Double> mipsShare = getCurrentMipsShare();
                if (mipsShare != null && !mipsShare.isEmpty()) {

                    // vmMips = GPU MFLOPS reference = VM.mips (e.g. 135,000 for TX2).
                    // VmSchedulerSpaceShared stores vm.getMips() per PE in the share list,
                    // so get(0) gives the single-channel GPU MFLOPS. Using sum() would give
                    // cores × gpu_mips (e.g. 2×135,000=270,000) — 2x too fast.
                    double vmMips = mipsShare.get(0);

                    if (vmMips > 0) {
                        // ── CPU channel ──────────────────────────────────────
                        // cloudletLength = CPU MI set by AITaskPropertyDual.computeCpuMI()
                        double cpuTime = (double) cl.getCloudletLength() / (double) cpuMips;

                        // ── GPU channel ──────────────────────────────────────
                        // tensors = GPU MFLOP; vmMips = GPU MFLOPS reference
                        double gpuTime = tensors / vmMips;

                        // ── Dual-channel wall-clock ──────────────────────────
                        double effectiveTime = Math.max(cpuTime, gpuTime);

                        // Rewrite cloudletLength so CloudSim formula gives effectiveTime:
                        //   processingTime = effectiveMI / vmMips = effectiveTime ✓
                        long effectiveMI = Math.max(1L,
                                Math.round(effectiveTime * vmMips));
                        cl.setCloudletLength(effectiveMI);  // ← setCloudletLength() confirmed in JAR
                    }
                }
            }
        }

        return super.cloudletSubmit(cl, fileTransferTime);
    }

    /** Returns the CPU MIPS configured for this scheduler. */
    public long getCpuMips() {
        return cpuMips;
    }
}
