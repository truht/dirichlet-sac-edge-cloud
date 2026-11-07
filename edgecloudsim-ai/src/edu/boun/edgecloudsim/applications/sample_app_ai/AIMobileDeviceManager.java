/*
 * Title:        EdgeCloudSim - AIMobileDeviceManager
 *
 * Description:
 * Mobile Device Manager for AI tasks. Extends the Tutorial3
 * SampleMobileDeviceManager pattern: handles WLAN/WAN/MAN paths
 * and adds GPU-specific logging (latency estimate, GPU load, RAM).
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.applications.sample_app_ai;

import org.cloudbus.cloudsim.UtilizationModel;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEvent;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.core.SimSettings.NETWORK_DELAY_TYPES;
import edu.boun.edgecloudsim.edge_client.CpuUtilizationModel_Custom;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_server.EdgeHost;
import edu.boun.edgecloudsim.edge_server.EdgeVM;
// gpu_ai imports
import edu.boun.edgecloudsim.gpu_ai.AITask;
import edu.boun.edgecloudsim.gpu_ai.AITaskProperty;
// gpu_dual imports (Phase 2)
import edu.boun.edgecloudsim.gpu_dual.AITaskPropertyDual;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.utils.Location;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.TaskProperty;

/**
 * AI-aware MobileDeviceManager.
 *
 * <p>Behaviour is identical to Tutorial3's SampleMobileDeviceManager for
 * network routing (WLAN/WAN/MAN relay paths). The key differences:
 * <ol>
 *   <li>Creates {@link AITask} instances (not plain Task) when the task
 *       property is an {@link AITaskProperty}.</li>
 *   <li>Logs GPU-specific metrics (estimated latency, GPU load, RAM) via
 *       SimLogger for post-simulation analysis.</li>
 * </ol>
 * </p>
 */
public class AIMobileDeviceManager extends MobileDeviceManager {

    private static final int BASE = 200000; // avoid tag collision with tutorial3
    private static final int UPDATE_MM1_QUEUE_MODEL   = BASE + 1;
    private static final int REQUEST_RECEIVED_BY_CLOUD          = BASE + 2;
    private static final int REQUEST_RECEIVED_BY_EDGE_DEVICE    = BASE + 3;
    private static final int REQUEST_RECEIVED_BY_REMOTE_EDGE    = BASE + 4;
    private static final int REQUEST_RECEIVED_BY_EDGE_TO_RELAY  = BASE + 5;
    private static final int RESPONSE_RECEIVED_BY_MOBILE        = BASE + 6;
    private static final int RESPONSE_EDGE_RELAY_MOBILE         = BASE + 7;

    private static final double MM1_UPDATE_INTERVAL = 5.0; // seconds

    private int taskIdCounter = 0;

    public AIMobileDeviceManager() throws Exception {}

    @Override public void initialize() {}

    @Override
    public UtilizationModel getCpuUtilizationModel() {
        return new CpuUtilizationModel_Custom();
    }

    @Override
    public void startEntity() {
        super.startEntity();
        schedule(getId(),
                SimSettings.CLIENT_ACTIVITY_START_TIME + MM1_UPDATE_INTERVAL,
                UPDATE_MM1_QUEUE_MODEL);
    }

    @Override
    protected void submitCloudlets() { /* driven by events */ }

    // =======================================================================
    // Task submission — creates AITask from AITaskProperty
    // =======================================================================

    @Override
    public void submitTask(TaskProperty edgeTask) {
        NetworkModel net = SimManager.getInstance().getNetworkModel();

        // Create task (AITask if property is AITaskProperty, else plain Task)
        Task task = createTask(edgeTask);

        Location loc = SimManager.getInstance().getMobilityModel()
                .getLocation(task.getMobileDeviceId(), CloudSim.clock());
        task.setSubmittedLocation(loc);

        SimLogger.getInstance().addLog(task.getMobileDeviceId(),
                task.getCloudletId(), task.getTaskType(),
                (int) task.getCloudletLength(),
                (int) task.getCloudletFileSize(),
                (int) task.getCloudletOutputSize());

        // Orchestrator decides destination
        long t0 = System.nanoTime();
        int destId = SimManager.getInstance().getEdgeOrchestrator().getDeviceToOffload(task);
        SimLogger.getInstance().setOrchestratorOverhead(task.getCloudletId(),
                System.nanoTime() - t0);

        // Determine upload path
        double delay;
        int nextEvent;
        NETWORK_DELAY_TYPES delayType;
        int networkDest;
        int vmTypeOrdinal;

        if (destId == SimSettings.CLOUD_DATACENTER_ID) {
            delay         = net.getUploadDelay(task.getMobileDeviceId(), SimSettings.CLOUD_DATACENTER_ID, task);
            nextEvent     = REQUEST_RECEIVED_BY_CLOUD;
            delayType     = NETWORK_DELAY_TYPES.WAN_DELAY;
            networkDest   = SimSettings.CLOUD_DATACENTER_ID;
            vmTypeOrdinal = SimSettings.VM_TYPES.CLOUD_VM.ordinal();
        } else {
            delay         = net.getUploadDelay(task.getMobileDeviceId(), SimSettings.GENERIC_EDGE_DEVICE_ID, task);
            nextEvent     = REQUEST_RECEIVED_BY_EDGE_DEVICE;
            delayType     = NETWORK_DELAY_TYPES.WLAN_DELAY;
            networkDest   = SimSettings.GENERIC_EDGE_DEVICE_ID;
            vmTypeOrdinal = SimSettings.VM_TYPES.EDGE_VM.ordinal();
        }

        if (delay > 0) {
            Vm selectedVM = SimManager.getInstance().getEdgeOrchestrator()
                    .getVmToOffload(task, destId);

            if (selectedVM != null) {
                task.setAssociatedDatacenterId(destId);
                task.setAssociatedHostId(selectedVM.getHost().getId());
                task.setAssociatedVmId(selectedVM.getId());

                // cloudletLength đã được đặt đúng trong AITaskProperty.computeGpuMFLOP()
                // VM.MIPS = GPU TFLOPS × 10^6 (MFLOPS) trong edge_devices.xml / config
                // → processingTime = MFLOP / MFLOPS = seconds  ✓ không cần tính lại

                getCloudletList().add(task);
                bindCloudletToVm(task.getCloudletId(), selectedVM.getId());

                // Detect MAN relay need (edge in different WLAN)
                if (selectedVM instanceof EdgeVM) {
                    EdgeHost host = (EdgeHost) selectedVM.getHost();
                    if (host.getLocation().getServingWlanId()
                            != task.getSubmittedLocation().getServingWlanId()) {
                        nextEvent = REQUEST_RECEIVED_BY_EDGE_TO_RELAY;
                    }
                }

                net.uploadStarted(loc, networkDest);
                SimLogger.getInstance().taskStarted(task.getCloudletId(), CloudSim.clock());
                SimLogger.getInstance().setUploadDelay(task.getCloudletId(), delay, delayType);
                schedule(getId(), delay, nextEvent, task);

                // Log AI-specific estimates
                // ĐÃ TẮT: in ~1M dòng ra log gây chậm nghiêm trọng (~37 phút/ite)
                // BẬT LẠI khi cần debug: bỏ comment đoạn bên dưới
                // if (task instanceof AITask) {
                //     AITask at = (AITask) task;
                //     SimLogger.printLine(String.format(
                //         "[AI] Task#%d %s → %s | estLatency=%.1fms gpuLoad=%.1f%% RAM=%.2fGB cloudletLen=%dMI",
                //         task.getCloudletId(),
                //         at.getCnnModelName(),
                //         destId == SimSettings.CLOUD_DATACENTER_ID ? "CLOUD" : "EDGE",
                //         at.getEstimatedLatencyMs(),
                //         at.getEstimatedGpuLoadPct(),
                //         at.getEstimatedRamGB(),
                //         at.getCloudletLength()));
                // }

            } else {
                SimLogger.getInstance().rejectedDueToVMCapacity(
                        task.getCloudletId(), CloudSim.clock(), vmTypeOrdinal);
            }
        } else {
            SimLogger.getInstance().rejectedDueToBandwidth(
                    task.getCloudletId(), CloudSim.clock(), vmTypeOrdinal, delayType);
        }
    }

    // =======================================================================
    // Cloudlet return — download path (identical to tutorial3)
    // =======================================================================

    @Override
    protected void processCloudletReturn(SimEvent ev) {
        NetworkModel net = SimManager.getInstance().getNetworkModel();
        Task task = (Task) ev.getData();
        SimLogger.getInstance().taskExecuted(task.getCloudletId());

        if (task.getAssociatedDatacenterId() == SimSettings.CLOUD_DATACENTER_ID) {
            double wanDelay = net.getDownloadDelay(SimSettings.CLOUD_DATACENTER_ID,
                    task.getMobileDeviceId(), task);
            if (wanDelay > 0) {
                Location futureLoc = SimManager.getInstance().getMobilityModel()
                        .getLocation(task.getMobileDeviceId(), CloudSim.clock() + wanDelay);
                if (task.getSubmittedLocation().getServingWlanId()
                        == futureLoc.getServingWlanId()) {
                    net.downloadStarted(task.getSubmittedLocation(),
                            SimSettings.CLOUD_DATACENTER_ID);
                    SimLogger.getInstance().setDownloadDelay(task.getCloudletId(),
                            wanDelay, NETWORK_DELAY_TYPES.WAN_DELAY);
                    schedule(getId(), wanDelay, RESPONSE_RECEIVED_BY_MOBILE, task);
                } else {
                    SimLogger.getInstance().failedDueToMobility(
                            task.getCloudletId(), CloudSim.clock());
                }
            } else {
                SimLogger.getInstance().failedDueToBandwidth(
                        task.getCloudletId(), CloudSim.clock(), NETWORK_DELAY_TYPES.WAN_DELAY);
            }
        } else {
            // Edge result path (with possible MAN relay back)
            int nextEvent   = RESPONSE_RECEIVED_BY_MOBILE;
            int netDest     = SimSettings.GENERIC_EDGE_DEVICE_ID;
            NETWORK_DELAY_TYPES dType = NETWORK_DELAY_TYPES.WLAN_DELAY;
            double delay = net.getDownloadDelay(task.getAssociatedHostId(),
                    task.getMobileDeviceId(), task);

            EdgeHost host = (EdgeHost) SimManager.getInstance()
                    .getEdgeServerManager()
                    .getDatacenterList().get(task.getAssociatedHostId())
                    .getHostList().get(0);

            if (host.getLocation().getServingWlanId()
                    != task.getSubmittedLocation().getServingWlanId()) {
                delay     = net.getDownloadDelay(SimSettings.GENERIC_EDGE_DEVICE_ID,
                        SimSettings.GENERIC_EDGE_DEVICE_ID, task);
                nextEvent = RESPONSE_EDGE_RELAY_MOBILE;
                netDest   = SimSettings.GENERIC_EDGE_DEVICE_ID + 1;
                dType     = NETWORK_DELAY_TYPES.MAN_DELAY;
            }

            if (delay > 0) {
                Location futureLoc = SimManager.getInstance().getMobilityModel()
                        .getLocation(task.getMobileDeviceId(), CloudSim.clock() + delay);
                if (task.getSubmittedLocation().getServingWlanId()
                        == futureLoc.getServingWlanId()) {
                    net.downloadStarted(futureLoc, netDest);
                    SimLogger.getInstance().setDownloadDelay(task.getCloudletId(), delay, dType);
                    schedule(getId(), delay, nextEvent, task);
                } else {
                    SimLogger.getInstance().failedDueToMobility(
                            task.getCloudletId(), CloudSim.clock());
                }
            } else {
                SimLogger.getInstance().failedDueToBandwidth(
                        task.getCloudletId(), CloudSim.clock(), dType);
            }
        }
    }

    // =======================================================================
    // Event dispatcher
    // =======================================================================

    @Override
    protected void processOtherEvent(SimEvent ev) {
        if (ev == null) { System.exit(1); return; }
        NetworkModel net = SimManager.getInstance().getNetworkModel();

        switch (ev.getTag()) {
            case UPDATE_MM1_QUEUE_MODEL:
                if (net instanceof AISampleNetworkModel) {
                    ((AISampleNetworkModel) net).updateMM1QueeuModel();
                }
                schedule(getId(), MM1_UPDATE_INTERVAL, UPDATE_MM1_QUEUE_MODEL);
                break;

            case REQUEST_RECEIVED_BY_CLOUD: {
                Task t = (Task) ev.getData();
                net.uploadFinished(t.getSubmittedLocation(), SimSettings.CLOUD_DATACENTER_ID);
                submitTaskToVm(t, SimSettings.VM_TYPES.CLOUD_VM);
                break;
            }
            case REQUEST_RECEIVED_BY_EDGE_DEVICE: {
                Task t = (Task) ev.getData();
                net.uploadFinished(t.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID);
                submitTaskToVm(t, SimSettings.VM_TYPES.EDGE_VM);
                break;
            }
            case REQUEST_RECEIVED_BY_REMOTE_EDGE: {
                Task t = (Task) ev.getData();
                net.uploadFinished(t.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID + 1);
                submitTaskToVm(t, SimSettings.VM_TYPES.EDGE_VM);
                break;
            }
            case REQUEST_RECEIVED_BY_EDGE_TO_RELAY: {
                Task t = (Task) ev.getData();
                net.uploadFinished(t.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID);
                double manDelay = net.getUploadDelay(SimSettings.GENERIC_EDGE_DEVICE_ID,
                        SimSettings.GENERIC_EDGE_DEVICE_ID, t);
                if (manDelay > 0) {
                    net.uploadStarted(t.getSubmittedLocation(),
                            SimSettings.GENERIC_EDGE_DEVICE_ID + 1);
                    SimLogger.getInstance().setUploadDelay(t.getCloudletId(), manDelay,
                            NETWORK_DELAY_TYPES.MAN_DELAY);
                    schedule(getId(), manDelay, REQUEST_RECEIVED_BY_REMOTE_EDGE, t);
                } else {
                    SimLogger.getInstance().rejectedDueToBandwidth(t.getCloudletId(),
                            CloudSim.clock(), SimSettings.VM_TYPES.EDGE_VM.ordinal(),
                            NETWORK_DELAY_TYPES.MAN_DELAY);
                }
                break;
            }
            case RESPONSE_EDGE_RELAY_MOBILE: {
                Task t = (Task) ev.getData();
                net.downloadFinished(t.getSubmittedLocation(),
                        SimSettings.GENERIC_EDGE_DEVICE_ID + 1);
                double wlanDelay = net.getDownloadDelay(t.getAssociatedHostId(),
                        t.getMobileDeviceId(), t);
                if (wlanDelay > 0) {
                    Location cur = SimManager.getInstance().getMobilityModel()
                            .getLocation(t.getMobileDeviceId(), CloudSim.clock() + wlanDelay);
                    if (t.getSubmittedLocation().getServingWlanId()
                            == cur.getServingWlanId()) {
                        net.downloadStarted(cur, SimSettings.GENERIC_EDGE_DEVICE_ID);
                        SimLogger.getInstance().setDownloadDelay(t.getCloudletId(),
                                wlanDelay, NETWORK_DELAY_TYPES.WLAN_DELAY);
                        schedule(getId(), wlanDelay, RESPONSE_RECEIVED_BY_MOBILE, t);
                    } else {
                        SimLogger.getInstance().failedDueToMobility(
                                t.getCloudletId(), CloudSim.clock());
                    }
                } else {
                    SimLogger.getInstance().failedDueToBandwidth(t.getCloudletId(),
                            CloudSim.clock(), NETWORK_DELAY_TYPES.WLAN_DELAY);
                }
                break;
            }
            case RESPONSE_RECEIVED_BY_MOBILE: {
                Task t = (Task) ev.getData();
                if (t.getAssociatedDatacenterId() == SimSettings.CLOUD_DATACENTER_ID) {
					net.downloadFinished(t.getSubmittedLocation(), SimSettings.CLOUD_DATACENTER_ID);
				} else {
					net.downloadFinished(t.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID);
				}
                SimLogger.getInstance().taskEnded(t.getCloudletId(), CloudSim.clock());
                break;
            }
            default:
                SimLogger.printLine("AIMobileDeviceManager: unknown event " + ev.getTag());
                System.exit(1);
        }
    }

    // =======================================================================
    // Helpers
    // =======================================================================

    private void submitTaskToVm(Task task, SimSettings.VM_TYPES vmType) {
        schedule(getVmsToDatacentersMap().get(task.getVmId()), 0,
                CloudSimTags.CLOUDLET_SUBMIT, task);
        SimLogger.getInstance().taskAssigned(task.getCloudletId(),
                task.getAssociatedDatacenterId(),
                task.getAssociatedHostId(),
                task.getAssociatedVmId(),
                vmType.ordinal());
    }

    private Task createTask(TaskProperty prop) {
        UtilizationModel uFull = new UtilizationModelFull();
        UtilizationModel uCpu  = getCpuUtilizationModel();

        // ── Phase 2: AITaskPropertyDual ──────────────────────────────────────
        // cloudletLength = CPU MI, gpuMFLOP separate → passed to AITask Phase-2 constructor
        // Must be checked BEFORE AITaskProperty (AITaskPropertyDual does NOT extend it)
        if (prop instanceof AITaskPropertyDual) {
            AITaskPropertyDual ap = (AITaskPropertyDual) prop;
            AITask aiTask = new AITask(
                    ap.getMobileDeviceId(), ++taskIdCounter,
                    ap.getLength(),          // CPU MI  (cloudletLength)
                    ap.getGpuMFLOP(),        // GPU MFLOP → task.tensors  ← Phase 2 key
                    ap.getPesNumber(),
                    ap.getInputFileSize(), ap.getOutputFileSize(),
                    uCpu, uFull, uFull,
                    ap.getCnnModel(),
                    ap.getBatchSize(),
                    ap.getConcurrentStreams(),
                    ap.getRequestRateHz(),
                    ap.getPreprocessTimeMs(),
                    ap.getPostprocessTimeMs(),
                    ap.getDutyCyclePct()
            );
            aiTask.setUserId(this.getId());
            aiTask.setTaskType(ap.getTaskType());
            if (uCpu instanceof CpuUtilizationModel_Custom) {
                ((CpuUtilizationModel_Custom) uCpu).setTask(aiTask);
            }
            return aiTask;
        }

        // ── Phase 1: AITaskProperty ───────────────────────────────────────
        // cloudletLength = GPU MFLOP (single-channel); task.tensors = same value
        if (prop instanceof AITaskProperty) {
            AITaskProperty ap = (AITaskProperty) prop;
            // Pass the full CNNModel object — AITask derives totalFLOPs from it
            AITask aiTask = new AITask(
                    ap.getMobileDeviceId(), ++taskIdCounter,
                    ap.getLength(), ap.getPesNumber(),
                    ap.getInputFileSize(), ap.getOutputFileSize(),
                    uCpu, uFull, uFull,
                    ap.getCnnModel(),           // ← full CNNModel object
                    ap.getBatchSize(),
                    ap.getConcurrentStreams(),
                    ap.getRequestRateHz(),
                    ap.getPreprocessTimeMs(),
                    ap.getPostprocessTimeMs(),
                    ap.getDutyCyclePct()
            );
            aiTask.setUserId(this.getId());
            aiTask.setTaskType(ap.getTaskType());
            if (uCpu instanceof CpuUtilizationModel_Custom) {
                ((CpuUtilizationModel_Custom) uCpu).setTask(aiTask);
            }
            return aiTask;
        }

        // Fallback: plain Task
        Task task = new Task(prop.getMobileDeviceId(), ++taskIdCounter,
                prop.getLength(), prop.getPesNumber(),
                prop.getInputFileSize(), prop.getOutputFileSize(),
                uCpu, uFull, uFull);
        task.setUserId(this.getId());
        task.setTaskType(prop.getTaskType());
        if (uCpu instanceof CpuUtilizationModel_Custom) {
            ((CpuUtilizationModel_Custom) uCpu).setTask(task);
        }
        return task;
    }

}
