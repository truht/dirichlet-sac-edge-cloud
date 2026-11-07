/*
 * Title:        EdgeCloudSim - DRL Scenario Factory
 *
 * Description:
 * ScenarioFactory implementation for the SAC-driven offloading scenario.
 * Built on the tutorial3_ai_flop track (gpu_dual / DualChannelScheduler,
 * GPU+CPU workload set directly in applications.xml) rather than the
 * gpu_ai/AITask/CNNModel track (sample_app_ai) -- tutorial3_ai_flop is
 * where GPU support was added in prior work, so this factory reuses its
 * components: FlopLoadGenerator, DualChannelEdgeServerManager,
 * DualChannelCloudServerManager, tutorial3.SampleNetworkModel. Only two
 * components differ, matching the "SAC replaces the orchestrator entirely"
 * decision:
 *   - getEdgeOrchestrator() -> DRLEdgeOrchestrator (Algorithm 2)
 *   - getMobileDeviceManager() -> DRLMobileDeviceManager (adds the
 *     VRAM release + latency-collection hook on cloudlet completion)
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.applications.drl_app;

import edu.boun.edgecloudsim.applications.tutorial3.SampleNetworkModel;
import edu.boun.edgecloudsim.applications.tutorial3_ai_flop.FlopLoadGenerator;
import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.drl.DRLEdgeOrchestrator;
import edu.boun.edgecloudsim.drl.DRLMobileDeviceManager;
import edu.boun.edgecloudsim.drl.DRLSocketBridge;
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

public class DRLScenarioFactory implements ScenarioFactory {

    private final int numOfMobileDevice;
    private final double simulationTime;
    private final String orchestratorPolicy;
    private final String simScenario;
    private final DRLSocketBridge bridge;
    private final int episode;

    /** Cached so DRLMainApp can inspect isShutdownRequested() after each episode. */
    private DRLEdgeOrchestrator orchestrator;

    public DRLScenarioFactory(int numOfMobileDevice, double simulationTime,
                               String orchestratorPolicy, String simScenario,
                               DRLSocketBridge bridge, int episode) {
        this.numOfMobileDevice  = numOfMobileDevice;
        this.simulationTime     = simulationTime;
        this.orchestratorPolicy = orchestratorPolicy;
        this.simScenario        = simScenario;
        this.bridge             = bridge;
        this.episode            = episode;
    }

    /** GPU/CPU workload set directly in applications.xml (<task_length>, <gpu_flops>, <gpu_vram_mb>). */
    @Override
    public LoadGeneratorModel getLoadGeneratorModel() {
        return new FlopLoadGenerator(numOfMobileDevice, simulationTime, simScenario);
    }

    @Override
    public EdgeOrchestrator getEdgeOrchestrator() {
        if (orchestrator == null) {
            orchestrator = new DRLEdgeOrchestrator(orchestratorPolicy, simScenario, bridge, episode);
        }
        return orchestrator;
    }

    /** Non-interface accessor: lets DRLMainApp check shutdown state after the episode ends. */
    public boolean wasShutdownRequested() {
        return orchestrator != null && orchestrator.isShutdownRequested();
    }

    @Override
    public MobilityModel getMobilityModel() {
        return new NomadicMobility(numOfMobileDevice, simulationTime);
    }

    /** Reused unchanged -- FlopMobileDeviceManager (DRLMobileDeviceManager's parent) needs this type. */
    @Override
    public NetworkModel getNetworkModel() {
        return new SampleNetworkModel(numOfMobileDevice, simScenario);
    }

    /** Dual-channel: creates DualChannelEdgeVm instances with VRAM admission control. */
    @Override
    public EdgeServerManager getEdgeServerManager() {
        return new DualChannelEdgeServerManager();
    }

    /** Dual-channel cloud VMs (plain CloudVM -- no VRAM tracking, Cloud stays the safety valve). */
    @Override
    public CloudServerManager getCloudServerManager() {
        return new DualChannelCloudServerManager();
    }

    @Override
    public MobileDeviceManager getMobileDeviceManager() throws Exception {
        return new DRLMobileDeviceManager();
    }

    @Override
    public MobileServerManager getMobileServerManager() {
        return new DefaultMobileServerManager();
    }
}
