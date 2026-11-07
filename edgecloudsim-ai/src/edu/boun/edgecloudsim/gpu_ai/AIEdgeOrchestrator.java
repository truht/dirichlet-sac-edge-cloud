/*
 * Title:        EdgeCloudSim - AIEdgeOrchestrator
 *
 * Description:
 * AI-aware edge orchestrator that makes offload decisions based on
 * GPU load, RAM availability, and latency constraints instead of
 * simple CPU utilisation thresholds.
 *
 * Supported policies:
 *   GPU_LOAD_BASED   — offload if GPU load > 90% or RAM insufficient
 *   LATENCY_BASED    — offload if estimated latency > SLA
 *   COMBINED         — offload if ANY constraint violated (recommended)
 *   RANDOM           — 50/50 baseline for comparison
 *
 * VM selection: Least-Loaded (Worst-Fit) — same as BasicEdgeOrchestrator
 * but also checks GPU RAM headroom.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.gpu_ai;

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
import edu.boun.edgecloudsim.edge_server.EdgeVM;
import edu.boun.edgecloudsim.gpu_ai.AIWorkloadCalculator.AIWorkloadResult;
import edu.boun.edgecloudsim.gpu_ai.AIWorkloadCalculator.OffloadDecision;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;

/**
 * AI-aware orchestrator extending {@link EdgeOrchestrator}.
 *
 * <p>When a task is an {@link AITask}, the orchestrator runs
 * {@link AIWorkloadCalculator#calculate} to estimate GPU load,
 * latency, and RAM requirements before deciding where to send it.
 * For non-AI tasks, falls back to standard CPU-utilisation logic.</p>
 *
 * <p>Supported policies (set in config orchestrator_policies):
 * <ul>
 *   <li>{@code GPU_LOAD_BASED} — offload when GPU &gt; 90% or OOM</li>
 *   <li>{@code LATENCY_BASED}  — offload when latency &gt; SLA</li>
 *   <li>{@code COMBINED}       — offload when ANY constraint violated</li>
 *   <li>{@code RANDOM}         — 50/50 for baseline comparison</li>
 * </ul>
 * </p>
 */
public class AIEdgeOrchestrator extends EdgeOrchestrator {

    private int numberOfHost;

    /** Edge device GPU profile used for AI calculations. */
    private final EdgeGPUDevice edgeGpuDevice;

    /**
     * Cache: model name → AIWorkloadResult
     *
     * MỤC ĐÍCH: tránh tính lại AIWorkloadCalculator.calculate() cho mỗi task.
     * Với ~1M tasks/ite nhưng chỉ 3 model type (ResNet-50, MobileNetV2, YOLOv5s),
     * cache giảm số lần tính từ ~1M xuống còn 3 → tăng tốc đáng kể.
     *
     * ĐIỀU KIỆN HỢP LỆ: tất cả task cùng model có cùng tham số
     * (batchSize, concurrentStreams, requestRateHz, preprocessMs, postprocessMs, dutyCyclePct).
     * Log xác nhận: ResNet-50 luôn cho 58.3ms/100%/0.65GB, MobileNetV2 luôn 10.0ms/6.7%/0.46GB.
     *
     * KHI NÀO CẦN BỎ CACHE:
     *   - GPU load động theo thời gian (GPU utilization thay đổi ảnh hưởng đến quyết định)
     *   - Batch size thay đổi per task (không cố định theo model type)
     *   - Muốn mô phỏng thermal throttling thực tế (dutyCyclePct thay đổi)
     * Khi đó: xóa 2 dòng cache bên dưới và dùng lại AIWorkloadCalculator.calculate() trực tiếp.
     */
    private final Map<String, AIWorkloadResult> workloadCache = new HashMap<>();

    public AIEdgeOrchestrator(String policy, String simScenario,
                               EdgeGPUDevice edgeGpuDevice) {
        super(policy, simScenario);
        this.edgeGpuDevice = edgeGpuDevice;
    }

    @Override
    public void initialize() {
        numberOfHost = SimSettings.getInstance().getNumOfEdgeHosts();
    }

    // =======================================================================
    // Core offload decision
    // =======================================================================

    /**
     * Determines where to offload the task: edge or cloud.
     *
     * <p>For AI tasks, uses {@link AIWorkloadCalculator} to evaluate
     * GPU constraints. For CPU tasks, falls back to edge utilisation check.</p>
     */
    @Override
    public int getDeviceToOffload(Task task) {

        // --- AI-aware path ---
        if (task instanceof AITask) {
            return getDeviceForAITask((AITask) task);
        }

        // --- Fallback: standard CPU utilisation check ---
        double edgeUtil = SimManager.getInstance()
                .getEdgeServerManager().getAvgUtilization();
        if (edgeUtil > 75.0) {
            return SimSettings.CLOUD_DATACENTER_ID;
        }
        return SimSettings.GENERIC_EDGE_DEVICE_ID;
    }

    private int getDeviceForAITask(AITask aiTask) {
        // Get the full CNN model directly from the task — no hardcoding needed
        CNNModel model = aiTask.getCnnModel();

        // ── CACHE WORKLOAD RESULT ──────────────────────────────────────────────
        // Lần đầu gặp model → tính và lưu vào cache.
        // Lần sau → lấy ngay từ cache, không tính lại.
        //
        // Để TẮT cache (khi cần GPU load động): thay đoạn này bằng:
        //   AIWorkloadResult result = AIWorkloadCalculator.calculate(
        //       model, edgeGpuDevice, aiTask.getBatchSize(),
        //       aiTask.getConcurrentStreams(), aiTask.getRequestRateHz(),
        //       aiTask.getPreprocessTimeMs(), aiTask.getPostprocessTimeMs(),
        //       aiTask.getDutyCyclePct());
        // ──────────────────────────────────────────────────────────────────────
        String cacheKey = model.getName();
        AIWorkloadResult result = workloadCache.computeIfAbsent(cacheKey, k ->
                AIWorkloadCalculator.calculate(
                        model,
                        edgeGpuDevice,
                        aiTask.getBatchSize(),
                        aiTask.getConcurrentStreams(),
                        aiTask.getRequestRateHz(),
                        aiTask.getPreprocessTimeMs(),
                        aiTask.getPostprocessTimeMs(),
                        aiTask.getDutyCyclePct()
                )
        );

        // Store estimates on task for logging
        aiTask.setEstimatedLatencyMs(result.getLatencyFinalMs());
        aiTask.setEstimatedRamGB(result.getRamGB());
        aiTask.setEstimatedGpuLoadPct(result.getGpuLoadPct());

        // Update simulated GPU state
        edgeGpuDevice.setCurrentGPUUtilPct(result.getGpuLoadPct());

        // --- Policy dispatch ---
        switch (policy) {
            case "GPU_LOAD_BASED":
                return decideByGpuLoad(result);

            case "LATENCY_BASED":
                return decideByLatency(result, aiTask);

            case "COMBINED": {
                OffloadDecision decision = AIWorkloadCalculator.decide(
                        result, edgeGpuDevice,
                        getSlaLatency(aiTask),
                        getMinAccuracy(aiTask),
                        model.getTop1Accuracy()
                );
                return decision.shouldOffload()
                        ? SimSettings.CLOUD_DATACENTER_ID
                        : SimSettings.GENERIC_EDGE_DEVICE_ID;
            }

            case "RANDOM":
                return SimUtils.getRandomDoubleNumber(0, 1) < 0.5
                        ? SimSettings.CLOUD_DATACENTER_ID
                        : SimSettings.GENERIC_EDGE_DEVICE_ID;

            default:
                SimLogger.printLine("AIEdgeOrchestrator: unknown policy '" + policy + "'. Terminating.");
                System.exit(1);
                return -1;
        }
    }

    private int decideByGpuLoad(AIWorkloadResult result) {
        boolean gpuOverload = result.getGpuLoadPct() >= 90.0;
        boolean oom         = result.getRamGB() > edgeGpuDevice.getFreeRAMGB();
        return (gpuOverload || oom)
                ? SimSettings.CLOUD_DATACENTER_ID
                : SimSettings.GENERIC_EDGE_DEVICE_ID;
    }

    private int decideByLatency(AIWorkloadResult result, AITask task) {
        double sla = getSlaLatency(task);
        return (result.getLatencyFinalMs() > sla)
                ? SimSettings.CLOUD_DATACENTER_ID
                : SimSettings.GENERIC_EDGE_DEVICE_ID;
    }

    // =======================================================================
    // VM selection — Least Loaded (Worst-Fit)
    // =======================================================================

    /**
     * Selects the VM with the most remaining CPU capacity that can accommodate the task.
     * Identical logic to BasicEdgeOrchestrator but applies to both cloud and edge.
     */
    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        Vm selectedVM = null;

        if (deviceId == SimSettings.CLOUD_DATACENTER_ID) {
            double selectedVmCapacity = 0;
            List<Host> list = SimManager.getInstance()
                    .getCloudServerManager().getDatacenter().getHostList();
            for (int h = 0; h < list.size(); h++) {
                List<CloudVM> vms = SimManager.getInstance()
                        .getCloudServerManager().getVmList(h);
                for (CloudVM vm : vms) {
                    double required = ((CpuUtilizationModel_Custom)
                            task.getUtilizationModelCpu())
                            .predictUtilization(vm.getVmType());
                    double available = 100.0 - vm.getCloudletScheduler()
                            .getTotalUtilizationOfCpu(CloudSim.clock());
                    if (required <= available && available > selectedVmCapacity) {
                        selectedVM = vm;
                        selectedVmCapacity = available;
                    }
                }
            }

        } else if (deviceId == SimSettings.GENERIC_EDGE_DEVICE_ID) {
            double selectedVmCapacity = 0;
            for (int h = 0; h < numberOfHost; h++) {
                List<EdgeVM> vms = SimManager.getInstance()
                        .getEdgeServerManager().getVmList(h);
                for (EdgeVM vm : vms) {
                    double required = ((CpuUtilizationModel_Custom)
                            task.getUtilizationModelCpu())
                            .predictUtilization(vm.getVmType());
                    double available = 100.0 - vm.getCloudletScheduler()
                            .getTotalUtilizationOfCpu(CloudSim.clock());
                    if (required <= available && available > selectedVmCapacity) {
                        selectedVM = vm;
                        selectedVmCapacity = available;
                    }
                }
            }

        } else {
            SimLogger.printLine("AIEdgeOrchestrator: unknown deviceId=" + deviceId);
            System.exit(1);
        }

        return selectedVM;
    }

    // =======================================================================
    // CloudSim entity lifecycle (no internal events needed)
    // =======================================================================

    @Override public void startEntity()    { /* nothing */ }
    @Override public void shutdownEntity() { /* nothing */ }
    @Override public void processEvent(SimEvent e) { /* nothing */ }

    /**
     * Returns the edge GPU device profile used by this orchestrator.
     * Called by AIMobileDeviceManager.computeGpuCloudletLength() to get
     * the correct TFLOPS for cloudletLength conversion.
     */
    public EdgeGPUDevice getEdgeGpuDevice() { return edgeGpuDevice; }

    // =======================================================================
    // Helpers
    // =======================================================================

    /** Extracts SLA latency from AITask or returns a generous default. */
    private double getSlaLatency(Task task) {
        if (task instanceof AITask) {
            // AITask doesn't carry SLA directly — look up via task type
            // Convention: use 100ms (real-time) as safe default
            return 100.0;
        }
        return 200.0;
    }

    private double getMinAccuracy(Task task) {
        return 60.0; // default minimum accuracy
    }
}
