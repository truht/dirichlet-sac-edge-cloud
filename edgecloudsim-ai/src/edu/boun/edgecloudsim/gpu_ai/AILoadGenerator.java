/*
 * Title:        EdgeCloudSim - AILoadGenerator
 *
 * Description:
 * Extends IdleActiveLoadGenerator to produce AI/CNN task properties.
 * Each generated task carries a CNNModel reference (with totalFLOPs)
 * and workload parameters in an AITaskProperty.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.gpu_ai;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.distribution.ExponentialDistribution;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;

/**
 * AI-aware load generator extending {@link LoadGeneratorModel} directly.
 *
 * <p>Key design: uses a {@code Map<appName, CNNModel>} to map each application
 * declared in {@code applications.xml} to its CNN model profile. This means
 * each task type (RESNET_INFERENCE, MOBILENET_INFERENCE, YOLO_DETECTION) gets
 * its correct model — not a round-robin assignment.</p>
 *
 * <p>We extend {@code LoadGeneratorModel} directly (not IdleActiveLoadGenerator)
 * because {@code IdleActiveLoadGenerator.taskTypeOfDevices} is package-private.</p>
 */
public class AILoadGenerator extends LoadGeneratorModel {

    /** Per-device task type assignment (mirrors IdleActiveLoadGenerator). */
    private int[] taskTypeOfDevices;

    /**
     * Map: application name (from applications.xml) → CNNModel.
     * e.g. "RESNET_INFERENCE" → CNNModel.RESNET50_FP32
     *      "MOBILENET_INFERENCE" → CNNModel.MOBILENETV2
     *      "YOLO_DETECTION" → CNNModel.YOLOV5S
     */
    private final Map<String, CNNModel> appModelMap;

    /** Workload parameters shared across all AI tasks. */
    private final int    batchSize;
    private final int    concurrentStreams;
    private final double dutyCyclePct;
    private final double minAccuracy;

    /**
     * Full constructor.
     *
     * @param numOfMobileDevices total mobile devices in simulation
     * @param simulationTime     total simulation duration (seconds)
     * @param simScenario        scenario name (passed to parent)
     * @param appModelMap        map: app name from applications.xml → CNNModel
     * @param batchSize          GPU batch size per inference
     * @param concurrentStreams  parallel camera/sensor streams per device
     * @param dutyCyclePct       GPU duty cycle ceiling (%)
     * @param minAccuracy        minimum model accuracy requirement (%)
     */
    public AILoadGenerator(int numOfMobileDevices, double simulationTime,
                           String simScenario,
                           Map<String, CNNModel> appModelMap,
                           int batchSize, int concurrentStreams,
                           double dutyCyclePct, double minAccuracy) {
        super(numOfMobileDevices, simulationTime, simScenario);

        this.appModelMap       = (appModelMap != null && !appModelMap.isEmpty())
                                 ? appModelMap : buildDefaultMap();
        this.batchSize         = batchSize;
        this.concurrentStreams = concurrentStreams;
        this.dutyCyclePct      = dutyCyclePct;
        this.minAccuracy       = minAccuracy;
    }

    /** Default map matching the 3 apps declared in applications.xml. */
    private static Map<String, CNNModel> buildDefaultMap() {
        Map<String, CNNModel> m = new HashMap<>();
        m.put("RESNET_INFERENCE",    CNNModel.RESNET50_FP32);
        m.put("MOBILENET_INFERENCE", CNNModel.MOBILENETV2);
        m.put("YOLO_DETECTION",      CNNModel.YOLOV5S);
        return m;
    }

    /**
     * Overrides initializeModel() to inject AITaskProperty instead of plain TaskProperty.
     *
     * <p>Each task type (from applications.xml) is resolved to a specific CNNModel
     * via appModelMap. Workload params (requestRateHz, preprocessMs, postprocessMs,
     * slaLatencyMs) are read directly from the lookup table columns set in the XML,
     * so each app type carries its own timing profile.</p>
     *
     * <p>Lookup table column mapping:
     * <pre>
     *   [2]  = poisson_interarrival  → 1/requestRateHz
     *   [3]  = active_period
     *   [4]  = idle_period
     *   [5]  = data_upload  (KB)     → input image size
     *   [6]  = data_download (KB)    → result size
     *   [8]  = required_core
     *   [12] = delay_sensitivity
     *   [13] = max_delay_requirement → slaLatencyMs (seconds → ms)
     * </pre>
     * </p>
     */
    @Override
    public void initializeModel() {
        taskList = new ArrayList<>();

        SimSettings ss = SimSettings.getInstance();
        double[][] lookup = ss.getTaskLookUpTable();
        int numTaskTypes  = lookup.length;

        // Build exponential RNG for each task type (same as parent)
        ExponentialDistribution[][] expRng =
                new ExponentialDistribution[numTaskTypes][3];
        for (int i = 0; i < numTaskTypes; i++) {
            if (lookup[i][0] == 0) {
				continue;
			}
            expRng[i][0] = new ExponentialDistribution(lookup[i][5]); // input size
            expRng[i][1] = new ExponentialDistribution(lookup[i][6]); // output size
            expRng[i][2] = new ExponentialDistribution(lookup[i][7]); // task length (symbolic)
        }

        taskTypeOfDevices = new int[numberOfMobileDevices];

        for (int devId = 0; devId < numberOfMobileDevices; devId++) {

            // Weighted random task type selection (same as parent)
            int taskType = -1;
            double selector = SimUtils.getRandomDoubleNumber(0, 100);
            double cumPct   = 0;
            for (int j = 0; j < numTaskTypes; j++) {
                cumPct += lookup[j][0];
                if (selector <= cumPct) { taskType = j; break; }
            }
            if (taskType == -1) {
                SimLogger.printLine("AILoadGenerator: no task type for device " + devId);
                continue;
            }
            taskTypeOfDevices[devId] = taskType;

            // *** Resolve CNNModel from app name in applications.xml ***
            String appName = ss.getTaskName(taskType);
            CNNModel model = appModelMap.getOrDefault(appName, CNNModel.RESNET50_FP32);
            if (!appModelMap.containsKey(appName)) {
                SimLogger.printLine("AILoadGenerator: no CNNModel mapping for app '"
                        + appName + "', using ResNet-50 FP32 as fallback.");
            }

            // Per-app workload params read from lookup table (set in applications.xml)
            double poissonMean   = lookup[taskType][2];  // = 1/requestRateHz
            double activePeriod  = lookup[taskType][3];
            double idlePeriod    = lookup[taskType][4];
            int    pesNumber     = (int) lookup[taskType][8];
            // col[13] = max_delay_requirement (seconds) → convert to ms for SLA
            double slaLatencyMs  = lookup[taskType][13] * 1000.0;
            if (slaLatencyMs <= 0) {
				slaLatencyMs = 100.0; // default 100ms
			}

            // requestRateHz derived from Poisson mean (inverse)
            double requestRateHz = (poissonMean > 0) ? (1.0 / poissonMean) : 30.0;

            // preprocessMs and postprocessMs: estimated from app type
            // YOLO_DETECTION gets 15ms NMS overhead; others get 8ms resize
            double preprocessMs  = 8.0;
            double postprocessMs = appName.contains("YOLO") ? 15.0 : 0.0;

            double activePeriodStart = SimUtils.getRandomDoubleNumber(
                    SimSettings.CLIENT_ACTIVITY_START_TIME,
                    SimSettings.CLIENT_ACTIVITY_START_TIME + activePeriod);
            double virtualTime = activePeriodStart;

            ExponentialDistribution rng = new ExponentialDistribution(poissonMean);

            while (virtualTime < simulationTime) {
                double interval = rng.sample();
                if (interval <= 0) {
					continue;
				}
                virtualTime += interval;

                if (virtualTime > activePeriodStart + activePeriod) {
                    activePeriodStart += activePeriod + idlePeriod;
                    virtualTime = activePeriodStart;
                    continue;
                }

                long inputSize  = (long) expRng[taskType][0].sample();
                long outputSize = (long) expRng[taskType][1].sample();

                // Create AITaskProperty with correct CNN model for this app type
                taskList.add(new AITaskProperty(
                        virtualTime, devId, taskType, pesNumber,
                        inputSize, outputSize,
                        model,                  // ← correct model per app name
                        batchSize, concurrentStreams, requestRateHz,
                        preprocessMs, postprocessMs,
                        dutyCyclePct,
                        slaLatencyMs, minAccuracy
                ));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Protected accessors for Phase-2 subclass (AILoadGeneratorDual)
    // -----------------------------------------------------------------------

    /** Returns the app-name → CNNModel map. */
    protected Map<String, CNNModel> getAppModelMap()   { return appModelMap; }

    /** Returns the GPU batch size. */
    protected int    getBatchSize()                    { return batchSize; }

    /** Returns the number of concurrent input streams per device. */
    protected int    getConcurrentStreams()             { return concurrentStreams; }

    /** Returns the GPU duty cycle ceiling (%). */
    protected double getDutyCyclePct()                 { return dutyCyclePct; }

    /** Returns the minimum acceptable model accuracy (%). */
    protected double getMinAccuracy()                  { return minAccuracy; }

    /**
     * Returns the assigned task type for the specified mobile device.
     * Required by {@link LoadGeneratorModel} abstract contract.
     */
    @Override
    public int getTaskTypeOfDevice(int deviceId) {
        if (taskTypeOfDevices == null || deviceId >= taskTypeOfDevices.length) {
			return 0;
		}
        return taskTypeOfDevices[deviceId];
    }
}
