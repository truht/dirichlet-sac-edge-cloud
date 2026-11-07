/*
 * Title:        EdgeCloudSim - tutorial3_ai_flop - SampleScenarioFactory
 *
 * Description:
 * Factory cho kịch bản TẢI AI — TÁCH BẠCH CPU và GPU, GPU đặt thẳng trong XML.
 *
 *   - cloudletLength (CPU MI)  = <task_length>  (kênh CPU)
 *   - tensors        (GPU MFLOP) = <gpu_flops>   (kênh GPU, đặt thẳng, không tính)
 *   - processingTime = max(cpuTime, gpuTime)     (DualChannelScheduler)
 *
 * Thành phần:
 *   - getLoadGeneratorModel : FlopLoadGenerator         (MỚI, Poisson, tách CPU/GPU)
 *   - getMobileDeviceManager: FlopMobileDeviceManager   (MỚI, nạp tensors)
 *   - getEdgeServerManager  : DualChannelEdgeServerManager (gpu_dual, max(cpu,gpu))
 *   - getCloudServerManager : DualChannelCloudServerManager (gpu_dual)
 *   - getEdgeOrchestrator   : tutorial3.SampleEdgeOrchestrator (TÁI DÙNG, không đổi)
 *   - getNetworkModel       : tutorial3.SampleNetworkModel     (TÁI DÙNG, không đổi)
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.applications.tutorial3_ai_flop;

import edu.boun.edgecloudsim.applications.tutorial3.SampleEdgeOrchestrator;
import edu.boun.edgecloudsim.applications.tutorial3.SampleNetworkModel;
import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.DefaultMobileServerManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileServerManager;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeServerManager;
import edu.boun.edgecloudsim.gpu_dual.DualChannelCloudServerManager;
import edu.boun.edgecloudsim.gpu_dual.DualChannelEdgeServerManager;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.mobility.NomadicMobility;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;

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

    /** MỚI: Poisson, tách CPU MI (task_length) và GPU MFLOP (gpu_flops). */
    @Override
    public LoadGeneratorModel getLoadGeneratorModel() {
        return new FlopLoadGenerator(numOfMobileDevice, simulationTime, simScenario);
    }

    /** TÁI DÙNG orchestrator tutorial3 (3 giải thuật, không đổi). */
    @Override
    public EdgeOrchestrator getEdgeOrchestrator() {
        return new SampleEdgeOrchestrator(orchestratorPolicy, simScenario);
    }

    @Override
    public MobilityModel getMobilityModel() {
        return new NomadicMobility(numOfMobileDevice, simulationTime);
    }

    /** TÁI DÙNG network model tutorial3 (FlopMobileDeviceManager cần kiểu này). */
    @Override
    public NetworkModel getNetworkModel() {
        return new SampleNetworkModel(numOfMobileDevice, simScenario);
    }

    /** Edge GPU dual-channel: VM.mips = gpu_mips, cpu_mips riêng; max(cpu,gpu). */
    @Override
    public EdgeServerManager getEdgeServerManager() {
        return new DualChannelEdgeServerManager();
    }

    /** Cloud GPU dual-channel. */
    @Override
    public CloudServerManager getCloudServerManager() {
        return new DualChannelCloudServerManager();
    }

    /** MỚI: nạp GPU MFLOP vào Task.tensors. */
    @Override
    public MobileDeviceManager getMobileDeviceManager() throws Exception {
        return new FlopMobileDeviceManager();
    }

    @Override
    public MobileServerManager getMobileServerManager() {
        return new DefaultMobileServerManager();
    }
}
