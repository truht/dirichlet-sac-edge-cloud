/*
 * Title:        EdgeCloudSim - Tutorial3-AI Phase 2 Main Application
 *
 * Description:
 * Entry point cho simulation Phase 2 — dual-channel GPU scheduling.
 *   processingTime = max( cpuTime, gpuTime )
 *
 * Khác biệt so với Phase 1 (tutorial3_ai/MainApp):
 *   Phase 1: SampleScenarioFactory  → cloudletLength = GPU MFLOP (single-channel)
 *   Phase 2: SampleScenarioFactoryDual → cloudletLength = CPU MI, tensors = GPU MFLOP
 *
 * Giải thuật (KHÔNG thay đổi):
 *   NETWORK_BASED   : offload lên cloud nếu WAN > 5 Mbps
 *   UTILIZATION_BASED: offload lên cloud nếu edge utilization > 75%
 *   RANDOM          : 50/50 coin flip
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.applications.tutorial3_ai_dual;

import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;

import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;

/**
 * Tutorial3-AI Phase 2: Dual-channel GPU scheduling.
 *
 * <p>Cấu trúc giống hệt tutorial3_ai/MainApp, chỉ khác ở APPLICATION_FOLDER
 * và SampleScenarioFactoryDual (inject DualChannelScheduler).</p>
 */
public class MainApp {

    public static final int    EXPECTED_NUM_OF_ARGS = 5;
    public static final String APPLICATION_FOLDER   = "tutorial3_ai_dual";

    public static void main(String[] args) {
        Log.disable();
        SimLogger.enablePrintLog();

        int iterationStart, iterationEnd;
        String configFile = null, outputFolder = null, edgeDevicesFile = null, applicationsFile = null;

        if (args.length == EXPECTED_NUM_OF_ARGS) {
            configFile       = args[0];
            edgeDevicesFile  = args[1];
            applicationsFile = args[2];
            outputFolder     = args[3];
            iterationStart   = Integer.parseInt(args[4]);
            iterationEnd     = iterationStart;
        } else {
            SimLogger.printLine("No CLI args — using defaults for IDE run.");
            configFile       = "scripts/" + APPLICATION_FOLDER + "/config/default_config.properties";
            applicationsFile = "scripts/" + APPLICATION_FOLDER + "/config/applications.xml";
            edgeDevicesFile  = "scripts/" + APPLICATION_FOLDER + "/config/edge_devices_tx2.xml";
            iterationStart   = 1;
            iterationEnd     = 10;
        }

        SimSettings SS = SimSettings.getInstance();
        if (!SS.initialize(configFile, edgeDevicesFile, applicationsFile)) {
            SimLogger.printLine("Cannot initialize simulation settings!");
            System.exit(0);
        }

        DateFormat df = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
        Date batchStart = Calendar.getInstance().getTime();
        SimLogger.printLine("Simulation started at " + df.format(batchStart));
        SimLogger.printLine("----------------------------------------------------------------------");

        for (int iter = iterationStart; iter <= iterationEnd; iter++) {

            if (args.length != EXPECTED_NUM_OF_ARGS) {
				outputFolder = "sim_results/" + APPLICATION_FOLDER + "/ite" + iter;
			}

            if (SS.getFileLoggingEnabled()) {
                SimLogger.enableFileLog();
                File dir = new File(outputFolder);
                if (dir.exists() && dir.isDirectory()) {
                    SimLogger.printLine("Output folder available; cleaning '" + outputFolder + "'");
                    for (File f : dir.listFiles()) {
						if (f.isFile() && !f.delete()) {
                            SimLogger.printLine("file cannot be deleted: " + f.getAbsolutePath());
                            System.exit(1);
                        }
					}
                } else {
                    SimLogger.printLine("Creating output folder '" + outputFolder + "'");
                    dir.mkdirs();
                }
            }

            for (int j = SS.getMinNumOfMobileDev();
                 j <= SS.getMaxNumOfMobileDev();
                 j += SS.getMobileDevCounterSize()) {

                for (int k = 0; k < SS.getSimulationScenarios().length; k++) {
                    for (int i = 0; i < SS.getOrchestratorPolicies().length; i++) {

                        String scenario = SS.getSimulationScenarios()[k];
                        String policy   = SS.getOrchestratorPolicies()[i];
                        Date scenStart  = Calendar.getInstance().getTime();

                        SimLogger.printLine("Scenario started at " + df.format(scenStart));
                        SimLogger.printLine("Scenario: " + scenario
                                + " - Policy: " + policy
                                + " - #iteration: " + iter);
                        SimLogger.printLine("Duration: "
                                + SS.getSimulationTime() / 60 + " min"
                                + " (warm up: " + SS.getWarmUpPeriod() / 60 + " min)"
                                + " - #devices: " + j);

                        SimLogger.getInstance().simStarted(outputFolder,
                                "SIMRESULT_T3AI_DUAL_" + scenario + "_" + policy
                                        + "_" + j + "DEVICES");

                        try {
                            CloudSim.init(2, Calendar.getInstance(), false, 0.01);

                            // Phase 2: SampleScenarioFactoryDual kết nối:
                            //   - SampleEdgeOrchestrator (tutorial3, KHÔNG thay đổi)
                            //   - AILoadGeneratorDual (gpu_dual, tải AI/CNN dual-channel)
                            //   - DualChannelEdgeServerManager (gpu_dual, edge VM + DualChannelScheduler)
                            //   - DualChannelCloudServerManager (gpu_dual, cloud VM + DualChannelScheduler)
                            //   - AIMobileDeviceManager (sample_app_ai, tái sử dụng)
                            ScenarioFactory factory = new SampleScenarioFactoryDual(
                                    j, SS.getSimulationTime(), policy, scenario);

                            SimManager manager = new SimManager(
                                    factory, j, scenario, policy);
                            manager.startSimulation();

                        } catch (Exception e) {
                            SimLogger.printLine("Simulation terminated due to unexpected error");
                            e.printStackTrace();
                            System.exit(0);
                        }

                        Date scenEnd = Calendar.getInstance().getTime();
                        SimLogger.printLine("Scenario finished at " + df.format(scenEnd)
                                + ". It took " + SimUtils.getTimeDifference(scenStart, scenEnd));
                        SimLogger.printLine("----------------------------------------------------------------------");
                    }
                }
            }
        }

        Date batchEnd = Calendar.getInstance().getTime();
        SimLogger.printLine("Simulation finished at " + df.format(batchEnd)
                + ". It took " + SimUtils.getTimeDifference(batchStart, batchEnd));
    }
}
