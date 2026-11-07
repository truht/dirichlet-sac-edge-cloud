/*
 * Title:        EdgeCloudSim - DualChannelSchedulerTimeShared  (Phase 2)
 *
 * Description:
 * TimeShared variant of DualChannelScheduler — for CLOUD VMs.
 *
 * Why TimeShared for cloud (same as DefaultCloudServerManager):
 *   CloudletSchedulerSpaceShared limits concurrency to numOfPEs tasks.
 *   Under NETWORK_BASED / UTILIZATION_BASED policies many tasks are routed
 *   to cloud; with SpaceShared they pile up in a waiting list.  Each CloudSim
 *   clock tick iterates the entire waiting list → O(n) per tick → simulation
 *   slows exponentially.  TimeShared has no waiting list — all tasks run
 *   concurrently sharing MIPS — so simulation stays O(1) per tick.
 *
 * The dual-channel max(cpuTime, gpuTime) override is identical to
 * DualChannelScheduler; only the parent class changes.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.gpu_dual;

import java.util.List;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.CloudletSchedulerTimeShared;
import org.cloudbus.cloudsim.ResCloudlet;

import edu.boun.edgecloudsim.edge_client.Task;

/**
 * TimeShared dual-channel cloudlet scheduler for cloud VMs (Phase 2).
 *
 * <p>Identical logic to {@link DualChannelScheduler} but extends
 * {@link CloudletSchedulerTimeShared} to avoid queue build-up when many
 * tasks are offloaded to the cloud datacenter.</p>
 */
public class DualChannelSchedulerTimeShared extends CloudletSchedulerTimeShared {

    private final long cpuMips;

    // Running-sum cache for getTotalUtilizationOfCpu(), which the base class
    // (and DRLEdgeOrchestrator.getVmToOffload(), called once per task per
    // candidate VM) recomputes by rescanning the full cloudletExecList on
    // every call -- O(backlog) per call, and the backlog itself grows
    // superlinearly once a VM is oversubscribed, compounding into the O(n^2)+
    // blowup measured going from 300->1500 devices (~5.4x tasks, >794x
    // wall-clock). Safe to cache as a running sum because
    // CpuUtilizationModel_Custom.getUtilization() ignores its `time`
    // argument and returns a fixed per-task-type value set once at task
    // creation (see edge_client/CpuUtilizationModel_Custom.java) -- so each
    // cloudlet's contribution never changes between admission and
    // completion. Only cloudletSubmit()/cloudletFinish() are live mutators
    // of cloudletExecList in this project (cloudletCancel/cloudletPause are
    // never called anywhere in this codebase); if that ever changes, this
    // cache must be updated at those call sites too.
    private double cachedCpuUtilization = 0.0;

    public DualChannelSchedulerTimeShared(long cpuMips) {
        super();
        this.cpuMips = cpuMips;
    }

    @Override
    public double cloudletSubmit(Cloudlet cl, double fileTransferTime) {

        if (cl instanceof Task) {
            Task task = (Task) cl;
            long tensors = task.getTensors();

            if (tensors > 0 && cpuMips > 0) {
                List<Double> mipsShare = getCurrentMipsShare();
                if (mipsShare != null && !mipsShare.isEmpty()) {
                    // get(0) = vm.getMips() = GPU MFLOPS (single channel).
                    // sum() would give cores × gpu_mips — incorrect.
                    double vmMips = mipsShare.get(0);

                    if (vmMips > 0) {
                        double cpuTime      = (double) cl.getCloudletLength() / (double) cpuMips;
                        double gpuTime      = tensors / vmMips;
                        double effectiveTime = Math.max(cpuTime, gpuTime);
                        long   effectiveMI   = Math.max(1L,
                                Math.round(effectiveTime * vmMips));
                        cl.setCloudletLength(effectiveMI);
                    }
                }
            }
        }

        double result = super.cloudletSubmit(cl, fileTransferTime);
        cachedCpuUtilization += cl.getUtilizationOfCpu(0);
        return result;
    }

    @Override
    public void cloudletFinish(ResCloudlet rcl) {
        cachedCpuUtilization -= rcl.getCloudlet().getUtilizationOfCpu(0);
        super.cloudletFinish(rcl);
    }

    @Override
    public double getTotalUtilizationOfCpu(double time) {
        return cachedCpuUtilization;
    }

    public long getCpuMips() { return cpuMips; }
}
