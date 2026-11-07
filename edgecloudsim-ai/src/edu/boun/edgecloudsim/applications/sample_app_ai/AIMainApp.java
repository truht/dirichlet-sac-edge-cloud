/*
 * Title:        EdgeCloudSim - AI Main Application
 *
 * Description:
 * Entry point for the AI/CNN offloading simulation.
 * Compares three GPU-aware orchestration policies:
 *   - GPU_LOAD_BASED  : offload when GPU > 90% or OOM
 *   - LATENCY_BASED   : offload when latency > SLA
 *   - COMBINED        : offload when ANY constraint violated
 *   - RANDOM          : 50/50 baseline
 *
 * Usage (CLI): config edgeDevices applications outputFolder iteration
 * Usage (IDE): runs with default scripts/sample_app_ai/config/
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.applications.sample_app_ai;

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
 * Main application for AI/CNN edge offloading simulation.
 *
 * <p>Structure mirrors Tutorial3/MainApp exactly to maintain consistency
 * with the rest of EdgeCloudSim. Only the ScenarioFactory differs.</p>
 */
public class AIMainApp {

    public static final int    EXPECTED_NUM_OF_ARGS = 5;
    public static final String APPLICATION_FOLDER   = "sample_app_ai";

    public static void main(String[] args) {
        Log.disable();
        SimLogger.enablePrintLog();

        int iterationStart, iterationEnd;
        String configFile, outputFolder, edgeDevicesFile, applicationsFile;

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
            edgeDevicesFile  = "scripts/" + APPLICATION_FOLDER + "/config/edge_devices.xml";
            int iteration    = 1;
            iterationStart   = iteration;
            iterationEnd     = iteration;
            outputFolder     = null; // set per iteration below
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
                    SimLogger.printLine("Output folder is available; cleaning '" + outputFolder + "'");
                    for (File f : dir.listFiles()) {
						if (f.isFile() && !f.delete()) {
                            SimLogger.printLine("file cannot be deleted: " + f.getAbsolutePath());
                            System.exit(1);
                        }
					}
                } else {
                    SimLogger.printLine("Output folder is not available; creating '" + outputFolder + "'");
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
                                + " (warm up period: " + SS.getWarmUpPeriod() / 60 + " min)"
                                + " - #devices: " + j);

                        SimLogger.getInstance().simStarted(outputFolder,
                                "SIMRESULT_AI_" + scenario + "_" + policy
                                        + "_" + j + "DEVICES");

                        try {
                            CloudSim.init(2, Calendar.getInstance(), false, 0.01);

                            ScenarioFactory factory = new AIScenarioFactory(
                                    j, SS.getSimulationTime(), policy, scenario);

                            SimManager manager = new SimManager(
                                    factory, j, scenario, policy);
                            manager.startSimulation();

                        } catch (Exception e) {
                            SimLogger.printLine("The simulation has been terminated due to an unexpected error");
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
