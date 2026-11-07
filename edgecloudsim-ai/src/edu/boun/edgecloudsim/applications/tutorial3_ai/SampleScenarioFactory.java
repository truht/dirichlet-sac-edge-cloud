/*
 * Title:        EdgeCloudSim - Tutorial3-AI Scenario Factory
 *
 * Description:
 * ScenarioFactory kết nối AI workload với 3 giải thuật offloading của Tutorial3.
 *
 * Thiết kế:
 *   - Giữ nguyên SampleEdgeOrchestrator (tutorial3) — 3 giải thuật không thay đổi
 *   - Thay IdleActiveLoadGenerator bằng AILoadGenerator — tải AI/CNN (GPU FLOPs)
 *   - Dùng AIMobileDeviceManager (sample_app_ai) — xử lý AITask, log GPU metrics
 *   - Dùng AISampleNetworkModel (sample_app_ai) — network WLAN/WAN/MAN
 *
 * Cơ chế mô phỏng GPU:
 *   cloudletLength (MFLOP) = CNNModel.effectiveFLOPs × archFactor × quantFactor × batchFactor / 10^6
 *   VM.MIPS (MFLOPS)       = GPU_TFLOPS × η × 10^6    (set trong edge_devices.xml)
 *   processingTime (s)     = cloudletLength / VM.MIPS  ← CloudSim tính tự động ✓
 *
 * Ví dụ Jetson TX2 + ResNet-50 FP32 batch=1 (sau calibration):
 *   cloudletLength = 7710M × 0.88(arch) × 1.0(quant) × 1.0(batch) × 1.50(overhead) / 10^6 = 10,178 MFLOP
 *   VM.MIPS        = 0.75 TFLOPS × 0.18 × 10^6 = 135,000 MFLOPS
 *   processingTime = 10178 / 135000 ≈ 0.0754s = 75ms ✓ (Bouzidi 2022 measured TX2)
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.applications.tutorial3_ai;

import java.util.HashMap;
import java.util.Map;

import edu.boun.edgecloudsim.applications.sample_app_ai.AIMobileDeviceManager;
import edu.boun.edgecloudsim.applications.sample_app_ai.AISampleNetworkModel;
// Giữ nguyên orchestrator tutorial3
import edu.boun.edgecloudsim.applications.tutorial3.SampleEdgeOrchestrator;
import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.cloud_server.DefaultCloudServerManager;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.DefaultMobileServerManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileServerManager;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeServerManager;
import edu.boun.edgecloudsim.gpu.GpuEdgeServerManager;
// AI components từ gpu_ai + sample_app_ai
import edu.boun.edgecloudsim.gpu_ai.AILoadGenerator;
import edu.boun.edgecloudsim.gpu_ai.CNNModel;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.mobility.NomadicMobility;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;

/**
 * Factory kết nối tutorial3 (orchestrator) + gpu_ai (load) + sample_app_ai (device manager).
 *
 * <p><b>Điểm mấu chốt — tại sao giải thuật tutorial3 hoạt động với AI task:</b></p>
 * <ul>
 *   <li>NETWORK_BASED: probe WAN delay → quyết định bằng bandwidth, độc lập với loại task ✓</li>
 *   <li>UTILIZATION_BASED: đọc VM CPU utilization. Vì VM.MIPS = GPU throughput,
 *       utilization phản ánh GPU load tự nhiên. Khi AI task chạy lâu (ResNet 50ms),
 *       VM bận → utilization cao → trigger offload → đúng hành vi GPU overload ✓</li>
 *   <li>RANDOM: 50/50 coin flip, hoàn toàn độc lập với task type ✓</li>
 * </ul>
 *
 * <p><b>Kết quả đánh giá:</b> So sánh 3 giải thuật về:
 * task failure rate, avg service time, avg processing time, offload ratio
 * trong điều kiện tải AI thực tế (GPU FLOPs-driven).</p>
 */
public class SampleScenarioFactory implements ScenarioFactory {

    private final int    numOfMobileDevice;
    private final double simulationTime;
    private final String orchestratorPolicy;
    private final String simScenario;

    public SampleScenarioFactory(int numOfMobileDevice, double simulationTime,
                                  String orchestratorPolicy, String simScenario) {
        this.numOfMobileDevice  = numOfMobileDevice;
        this.simulationTime     = simulationTime;
        this.orchestratorPolicy = orchestratorPolicy;
        this.simScenario        = simScenario;
    }

    /**
     * AI Load Generator — tạo AITask với CNNModel thay plain Task.
     *
     * <p>Mỗi app trong applications.xml được map tới một CNN model cụ thể.
     * cloudletLength của AITask = GPU FLOPs đã được hiệu chỉnh (MFLOP).</p>
     */
    @Override
    public LoadGeneratorModel getLoadGeneratorModel() {
        // Map app name (applications.xml) → CNN model profile
        // Tên phải khớp chính xác với <application name="..."> trong applications.xml
        Map<String, CNNModel> appModelMap = new HashMap<>();
        appModelMap.put("RESNET_INFERENCE",    CNNModel.RESNET50_FP32);
        appModelMap.put("MOBILENET_INFERENCE", CNNModel.MOBILENETV2);
        appModelMap.put("YOLO_DETECTION",      CNNModel.YOLOV5S);

        return new AILoadGenerator(
                numOfMobileDevice,
                simulationTime,
                simScenario,
                appModelMap,
                1,     // batchSize = 1 (real-time single-image inference)
                1,     // concurrentStreams per device
                75.0,  // dutyCyclePct — safe thermal headroom cho Jetson
                70.0   // minAccuracy % — SLA accuracy threshold
        );
    }

    /**
     * SampleEdgeOrchestrator từ tutorial3 — KHÔNG thay đổi một dòng code nào.
     *
     * <p>3 policies:
     * <ul>
     *   <li>NETWORK_BASED: offload nếu WAN > 5 Mbps</li>
     *   <li>UTILIZATION_BASED: offload nếu edge utilization > 75%</li>
     *   <li>RANDOM: 50/50 coin flip</li>
     * </ul>
     * VM selection: Least Loaded (worst fit) cho cả edge và cloud.</p>
     */
    @Override
    public EdgeOrchestrator getEdgeOrchestrator() {
        // Import trực tiếp từ tutorial3 — zero code change
        return new SampleEdgeOrchestrator(orchestratorPolicy, simScenario);
    }

    @Override
    public MobilityModel getMobilityModel() {
        return new NomadicMobility(numOfMobileDevice, simulationTime);
    }

    /**
     * AISampleNetworkModel — mô hình mạng empirical WLAN/WAN + MM1 MAN.
     * Tái sử dụng từ sample_app_ai, tương thích với AITask.
     */
    @Override
    public NetworkModel getNetworkModel() {
        return new AISampleNetworkModel(numOfMobileDevice, simScenario);
    }

    @Override
    public EdgeServerManager getEdgeServerManager() {
        // GpuEdgeServerManager tạo GpuVm (SpaceShared scheduler + VRAM tracking)
        // thay vì EdgeVM (TimeShared) của DefaultEdgeServerManager.
        // Đọc gpu_mips và vram_mb từ edge_devices.xml.
        return new GpuEdgeServerManager();
    }

    @Override
    public CloudServerManager getCloudServerManager() {
        // Cloud GPU (T4-class) — VM.MIPS từ mips_for_cloud_vm trong config
        return new DefaultCloudServerManager();
    }

    /**
     * AIMobileDeviceManager — xử lý AITask, hỗ trợ WLAN/WAN/MAN routing.
     * Log GPU-specific metrics (latency, GPU load, RAM) vào SimLogger.
     */
    @Override
    public MobileDeviceManager getMobileDeviceManager() throws Exception {
        return new AIMobileDeviceManager();
    }

    @Override
    public MobileServerManager getMobileServerManager() {
        // Mobile GPU disabled — tasks go edge hoặc cloud
        return new DefaultMobileServerManager();
    }
}
