/*
 * Title:        EdgeCloudSim - Tutorial3-AI Phase 2 Scenario Factory
 *
 * Description:
 * Phase 2 of tutorial3_ai — dual-channel GPU scheduling.
 *   processingTime = max( cpuTime, gpuTime )
 *
 * Khác biệt so với Phase 1 (tutorial3_ai):
 *   Phase 1 (gpu package):
 *     cloudletLength = GPU MFLOP  (single-channel)
 *     GpuCloudletScheduler: SpaceShared, không override cloudletSubmit()
 *
 *   Phase 2 (gpu_dual package):
 *     cloudletLength = CPU MI     (preprocessing only)
 *     task.tensors   = GPU MFLOP  (inference)
 *     DualChannelScheduler: override cloudletSubmit() → setCloudletLength(max(cpu,gpu))
 *
 * Verification ResNet-50 batch=1:
 *   cpuTime =    24 MI   /   3,000 MIPS   =   8 ms  (ARM Cortex-A57)
 *   gpuTime = 10,178 MFLOP / 135,000 MFLOPS = 75 ms  (TX2 GPU)
 *   effectiveTime = max(8, 75) = 75 ms  ✓
 *
 *   Cloud T4:
 *   cpuTime =    24 MI   / 100,000 MIPS   =  0.24 ms
 *   gpuTime = 10,178 MFLOP / 3,500,000   =  2.9  ms  ✓
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.applications.tutorial3_ai_dual;

import java.util.HashMap;
import java.util.Map;

// Phase 1 components (dùng lại AIMobileDeviceManager + AISampleNetworkModel)
import edu.boun.edgecloudsim.applications.sample_app_ai.AIMobileDeviceManager;
import edu.boun.edgecloudsim.applications.sample_app_ai.AISampleNetworkModel;
// Orchestrator từ tutorial3 (KHÔNG thay đổi)
import edu.boun.edgecloudsim.applications.tutorial3.SampleEdgeOrchestrator;
import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.DefaultMobileServerManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileServerManager;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeServerManager;
// Phase 2 components (gpu_dual package)
import edu.boun.edgecloudsim.gpu_ai.CNNModel;
import edu.boun.edgecloudsim.gpu_dual.AILoadGeneratorDual;
import edu.boun.edgecloudsim.gpu_dual.DualChannelCloudServerManager;
import edu.boun.edgecloudsim.gpu_dual.DualChannelEdgeServerManager;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.mobility.NomadicMobility;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;

/**
 * Phase 2 ScenarioFactory.
 *
 * <p>Wires Phase 2 components:
 * <ul>
 *   <li>{@link AILoadGeneratorDual} — produces {@code AITaskPropertyDual}
 *       (cloudletLength = CPU MI, tensors = GPU MFLOP)</li>
 *   <li>{@link DualChannelEdgeServerManager} — edge VMs with
 *       {@code DualChannelScheduler} (reads cpu_mips from XML)</li>
 *   <li>{@link DualChannelCloudServerManager} — cloud VMs with
 *       {@code DualChannelScheduler} (cpuMips = 100,000)</li>
 *   <li>{@link AIMobileDeviceManager} — reused, handles both Phase 1 and 2
 *       via instanceof dispatch in createTask()</li>
 *   <li>{@link SampleEdgeOrchestrator} — UNCHANGED 3 algorithms</li>
 * </ul>
 * </p>
 */
public class SampleScenarioFactoryDual implements ScenarioFactory {

    private final int    numOfMobileDevice;
    private final double simulationTime;
    private final String orchestratorPolicy;
    private final String simScenario;

    public SampleScenarioFactoryDual(int numOfMobileDevice, double simulationTime,
                                      String orchestratorPolicy, String simScenario) {
        this.numOfMobileDevice  = numOfMobileDevice;
        this.simulationTime     = simulationTime;
        this.orchestratorPolicy = orchestratorPolicy;
        this.simScenario        = simScenario;
    }

    @Override
    public LoadGeneratorModel getLoadGeneratorModel() {
        Map<String, CNNModel> appModelMap = new HashMap<>();
        appModelMap.put("RESNET_INFERENCE",    CNNModel.RESNET50_FP32);
        appModelMap.put("MOBILENET_INFERENCE", CNNModel.MOBILENETV2);
        appModelMap.put("YOLO_DETECTION",      CNNModel.YOLOV5S);

        // AILoadGeneratorDual: tạo AITaskPropertyDual thay vì AITaskProperty
        return new AILoadGeneratorDual(
                numOfMobileDevice,
                simulationTime,
                simScenario,
                appModelMap,
                1,     // batchSize
                1,     // concurrentStreams
                75.0,  // dutyCyclePct
                70.0   // minAccuracy
        );
    }

    @Override
    public EdgeOrchestrator getEdgeOrchestrator() {
        // Giữ nguyên 3 giải thuật tutorial3 — zero change
        return new SampleEdgeOrchestrator(orchestratorPolicy, simScenario);
    }

    @Override
    public MobilityModel getMobilityModel() {
        return new NomadicMobility(numOfMobileDevice, simulationTime);
    }

    @Override
    public NetworkModel getNetworkModel() {
        return new AISampleNetworkModel(numOfMobileDevice, simScenario);
    }

    @Override
    public EdgeServerManager getEdgeServerManager() {
        // DualChannelEdgeServerManager: tạo DualChannelEdgeVm với DualChannelScheduler
        // Đọc cpu_mips (3000) và gpu_mips (135000) từ edge_devices_tx2.xml
        return new DualChannelEdgeServerManager();
    }

    @Override
    public CloudServerManager getCloudServerManager() {
        // DualChannelCloudServerManager: cloud VMs cũng dùng DualChannelScheduler
        // cpuMips = 100,000 (Intel Xeon), vmMips = 3,500,000 (T4 GPU)
        return new DualChannelCloudServerManager();
    }

    @Override
    public MobileDeviceManager getMobileDeviceManager() throws Exception {
        // AIMobileDeviceManager được tái sử dụng — đã có instanceof AITaskPropertyDual branch
        return new AIMobileDeviceManager();
    }

    @Override
    public MobileServerManager getMobileServerManager() {
        return new DefaultMobileServerManager();
    }
}
