/*
 * Title:        EdgeCloudSim - DRLMobileDeviceManager
 *
 * Description:
 * Thin subclass of FlopMobileDeviceManager (tutorial3_ai_flop track --
 * GPU/CPU workload set directly in applications.xml via DualChannelScheduler,
 * see /PROTOCOL.md for why this track was chosen over gpu_ai/AITask).
 * Behaviour is 100% identical to FlopMobileDeviceManager except that it
 * also notifies DRLEdgeOrchestrator when a cloudlet finishes, so the
 * orchestrator can release the VRAM it reserved on the specific
 * DualChannelEdgeVm the task ran on and record the task's latency for
 * that time-slot's CVaR calculation.
 *
 * NOTE: latency is measured at cloudlet-return time (CloudSim.clock()
 * minus task creation time), i.e. upload + queueing + execution -- it
 * does NOT include the download/response network delay, which
 * FlopMobileDeviceManager applies afterward via its own private event
 * dispatch. Same tradeoff as the previous AIMobileDeviceManager-based
 * version: download delay is small relative to upload+queue+exec and
 * applies roughly uniformly, so it doesn't distort the *relative* CVaR
 * signal SAC trains on, and reimplementing the private event switch
 * just to measure it isn't worth the risk.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.drl;

import java.util.List;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;

import edu.boun.edgecloudsim.applications.tutorial3_ai_flop.FlopMobileDeviceManager;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeVM;
import edu.boun.edgecloudsim.gpu_dual.DualChannelEdgeVm;

public class DRLMobileDeviceManager extends FlopMobileDeviceManager {

    public DRLMobileDeviceManager() throws Exception {
        super();
    }

    @Override
    protected void processCloudletReturn(SimEvent ev) {
        Task task = (Task) ev.getData();
        double latencyMs = (CloudSim.clock() - task.getCreationTime()) * 1000.0;

        EdgeOrchestrator orchestrator = SimManager.getInstance().getEdgeOrchestrator();
        if (orchestrator instanceof DRLEdgeOrchestrator) {
            boolean executedOnEdge = task.getAssociatedDatacenterId() != SimSettings.CLOUD_DATACENTER_ID;
            if (executedOnEdge) {
                releaseVram(task);
            }
            ((DRLEdgeOrchestrator) orchestrator).onTaskCompleted(
                    task, task.getAssociatedHostId(), latencyMs, executedOnEdge);
        }

        super.processCloudletReturn(ev);
    }

    /** Finds the exact DualChannelEdgeVm the task ran on and releases its VRAM hold. */
    private void releaseVram(Task task) {
        List<EdgeVM> vms = SimManager.getInstance().getEdgeServerManager()
                .getVmList(task.getAssociatedHostId());
        for (EdgeVM vm : vms) {
            if (vm.getId() == task.getAssociatedVmId() && vm instanceof DualChannelEdgeVm) {
                ((DualChannelEdgeVm) vm).releaseVram(task.getVramRequired());
                return;
            }
        }
    }
}
