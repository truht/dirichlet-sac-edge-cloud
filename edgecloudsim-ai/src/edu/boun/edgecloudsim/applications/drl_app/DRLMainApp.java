/*
 * Title:        EdgeCloudSim - DRL Main Application
 *
 * Description:
 * Entry point for the SAC-driven offloading simulation (Algorithm 1 +
 * Algorithm 2, see /PROTOCOL.md and /introduction.md). Opens the
 * DRLSocketBridge exactly once, before the episode loop -- matching
 * Algorithm 1 step 4, which establishes the socket connection before
 * "FOR episode = 1 TO Max_Episodes" -- then runs one fresh CloudSim
 * simulation (= one SAC episode) per iteration, reusing the same bridge
 * across all of them.
 *
 * Usage (CLI): config edgeDevices applications outputFolder
 * Usage (IDE): runs with default scripts/drl_app/config/
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.applications.drl_app;

import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.drl.DRLSocketBridge;
import edu.boun.edgecloudsim.utils.SimLogger;

public class DRLMainApp {

    public static final int    EXPECTED_NUM_OF_ARGS = 4;
    public static final String APPLICATION_FOLDER   = "drl_app";
    // NOTE: port 5000 conflicts with macOS ControlCenter/AirPlay Receiver,
    // which binds it by default -- verified during smoke testing on this
    // machine. 6000 avoids that collision; keep in sync with
    // drl_controller/config.py's Config.port.
    public static final int    SOCKET_PORT           = 6000;

    /**
     * Max_Episodes is NOT sent over the socket (see PROTOCOL.md section 4.4)
     * -- keep this in sync with drl_controller/config.py's Config.max_episodes.
     */
    public static final int MAX_EPISODES = 5000;

    public static void main(String[] args) throws Exception {
        Log.disable();
        SimLogger.enablePrintLog();

        String configFile, outputFolder, edgeDevicesFile, applicationsFile;

        if (args.length == EXPECTED_NUM_OF_ARGS) {
            configFile       = args[0];
            edgeDevicesFile  = args[1];
            applicationsFile = args[2];
            outputFolder     = args[3];
        } else {
            SimLogger.printLine("No CLI args — using defaults for IDE run.");
            configFile       = "scripts/" + APPLICATION_FOLDER + "/config/default_config.properties";
            applicationsFile = "scripts/" + APPLICATION_FOLDER + "/config/applications.xml";
            edgeDevicesFile  = "scripts/" + APPLICATION_FOLDER + "/config/edge_devices.xml";
            outputFolder     = "sim_results/" + APPLICATION_FOLDER;
        }

        SimSettings SS = SimSettings.getInstance();
        if (!SS.initialize(configFile, edgeDevicesFile, applicationsFile)) {
            SimLogger.printLine("Cannot initialize simulation settings!");
            System.exit(0);
        }

        int numOfMobileDevice = SS.getMinNumOfMobileDev();
        String[] scenarios = SS.getSimulationScenarios();
        String scenario = scenarios.length > 0 ? scenarios[0] : "SAC_SCENARIO";

        DateFormat df = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
        Date batchStart = Calendar.getInstance().getTime();
        SimLogger.printLine("DRL training started at " + df.format(batchStart));
        SimLogger.printLine("Duration per episode: " + SS.getSimulationTime() / 60 + " min"
                + " (warm up: " + SS.getWarmUpPeriod() / 60 + " min)"
                + " - #devices: " + numOfMobileDevice
                + " - Max_Episodes: " + MAX_EPISODES);
        SimLogger.printLine("----------------------------------------------------------------------");

        DRLSocketBridge bridge = new DRLSocketBridge(SOCKET_PORT);
        bridge.waitForClient();

        int episode = 1;
        try {
            for (; episode <= MAX_EPISODES; episode++) {
                String episodeOutputFolder = outputFolder + "/ep" + episode;
                if (SS.getFileLoggingEnabled()) {
                    SimLogger.enableFileLog();
                    File dir = new File(episodeOutputFolder);
                    if (!dir.exists()) {
                        dir.mkdirs();
                    }
                }
                SimLogger.getInstance().simStarted(episodeOutputFolder, "SIMRESULT_DRL_ep" + episode);

                CloudSim.init(2, Calendar.getInstance(), false, 0.01);
                DRLScenarioFactory factory = new DRLScenarioFactory(
                        numOfMobileDevice, SS.getSimulationTime(), "SAC", scenario, bridge, episode);
                SimManager manager = new SimManager(factory, numOfMobileDevice, scenario, "SAC");
                manager.startSimulation();

                if (episode == 1 || episode % 10 == 0) {
                    SimLogger.printLine("Episode " + episode + "/" + MAX_EPISODES + " finished.");
                }

                if (factory.wasShutdownRequested()) {
                    SimLogger.printLine("DRL controller requested an early stop after episode " + episode + ".");
                    break;
                }
            }
        } catch (Exception e) {
            SimLogger.printLine("DRL training terminated due to an unexpected error at episode " + episode);
            e.printStackTrace();
        } finally {
            bridge.close();
        }

        Date batchEnd = Calendar.getInstance().getTime();
        SimLogger.printLine("----------------------------------------------------------------------");
        SimLogger.printLine("DRL training finished at " + df.format(batchEnd)
                + " after " + Math.min(episode, MAX_EPISODES) + " episode(s).");
    }
}
