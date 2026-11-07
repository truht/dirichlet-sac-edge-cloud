/*
 * Title:        EdgeCloudSim - AI Scenario Factory
 *
 * Description:
 * ScenarioFactory implementation for AI/CNN offloading simulation.
 * Wires in the gpu_ai package components without changing any
 * existing EdgeCloudSim infrastructure.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.applications.sample_app_ai;

import java.util.HashMap;
import java.util.Map;

import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.cloud_server.DefaultCloudServerManager;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.DefaultMobileServerManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileServerManager;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.DefaultEdgeServerManager;
import edu.boun.edgecloudsim.edge_server.EdgeServerManager;
// gpu_ai package imports
import edu.boun.edgecloudsim.gpu_ai.AIEdgeOrchestrator;
import edu.boun.edgecloudsim.gpu_ai.AILoadGenerator;
import edu.boun.edgecloudsim.gpu_ai.CNNModel;
import edu.boun.edgecloudsim.gpu_ai.EdgeGPUDevice;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.mobility.NomadicMobility;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;

/**
 * Factory for the AI offloading simulation scenario.
 *
 * <p>Key components injected:
 * <ul>
 *   <li>{@link AILoadGenerator} — generates AI tasks with CNN model descriptors</li>
 *   <li>{@link AIEdgeOrchestrator} — GPU-aware offload decision engine</li>
 *   <li>{@link AISampleNetworkModel} — reuses tutorial3's empirical WLAN/WAN model</li>
 *   <li>All infrastructure managers: Default implementations (no changes needed)</li>
 * </ul>
 * </p>
 */
public class AIScenarioFactory implements ScenarioFactory {

    private final int    numOfMobileDevice;
    private final double simulationTime;
    private final String orchestratorPolicy;
    private final String simScenario;

    // GPU device profile — determines which Jetson hardware to simulate
    private final EdgeGPUDevice gpuDevice;

    public AIScenarioFactory(int numOfMobileDevice, double simulationTime,
                              String orchestratorPolicy, String simScenario) {
        this.numOfMobileDevice  = numOfMobileDevice;
        this.simulationTime     = simulationTime;
        this.orchestratorPolicy = orchestratorPolicy;
        this.simScenario        = simScenario;

        // Select GPU device based on scenario name (configurable)
        // Convention: scenario name contains device hint
        if (simScenario.contains("NANO")) {
            this.gpuDevice = EdgeGPUDevice.JETSON_NANO;
        } else if (simScenario.contains("AGX")) {
            this.gpuDevice = EdgeGPUDevice.JETSON_AGX;
        } else {
            this.gpuDevice = EdgeGPUDevice.JETSON_TX2; // default
        }
    }

    @Override
    public LoadGeneratorModel getLoadGeneratorModel() {
        // Map: application name (applications.xml) → CNNModel profile
        // Must match the <application name="..."> tags exactly
        Map<String, CNNModel> appModelMap = new HashMap<>();
        appModelMap.put("RESNET_INFERENCE",    CNNModel.RESNET50_FP32);
        appModelMap.put("MOBILENET_INFERENCE", CNNModel.MOBILENETV2);
        appModelMap.put("YOLO_DETECTION",      CNNModel.YOLOV5S);

        return new AILoadGenerator(
                numOfMobileDevice, simulationTime, simScenario,
                appModelMap,
                1,    // batchSize = 1 (real-time single-image inference)
                1,    // concurrentStreams per device
                75.0, // dutyCyclePct — safe thermal operating point
                70.0  // minAccuracy % — minimum acceptable model accuracy
                // requestRateHz, preprocessMs, postprocessMs, slaLatencyMs
                // are now read per-app from applications.xml lookup table
        );
    }

    @Override
    public EdgeOrchestrator getEdgeOrchestrator() {
        return new AIEdgeOrchestrator(orchestratorPolicy, simScenario, gpuDevice);
    }

    @Override
    public MobilityModel getMobilityModel() {
        return new NomadicMobility(numOfMobileDevice, simulationTime);
    }

    @Override
    public NetworkModel getNetworkModel() {
        // Reuse Tutorial3's empirical WLAN/WAN + MM1 MAN model
        return new AISampleNetworkModel(numOfMobileDevice, simScenario);
    }

    @Override
    public EdgeServerManager getEdgeServerManager() {
        return new DefaultEdgeServerManager();
    }

    @Override
    public CloudServerManager getCloudServerManager() {
        return new DefaultCloudServerManager();
    }

    @Override
    public MobileDeviceManager getMobileDeviceManager() throws Exception {
        return new AIMobileDeviceManager();
    }

    @Override
    public MobileServerManager getMobileServerManager() {
        // Mobile GPU disabled — tasks go to edge or cloud
        return new DefaultMobileServerManager();
    }
}
