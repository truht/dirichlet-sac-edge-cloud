/*
 * Title:        EdgeCloudSim - AILoadGeneratorDual  (Phase 2)
 *
 * Description:
 * Produces AITaskPropertyDual entries (instead of AITaskProperty).
 * The only difference: taskList.add(new AITaskPropertyDual(...)) instead
 * of taskList.add(new AITaskProperty(...)).
 *
 * AITaskPropertyDual sets:
 *   cloudletLength = CPU MI   (preprocessing)
 *   gpuMFLOP       = GPU MFLOP (stored separately, passed to task.tensors)
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.gpu_dual;

import java.util.ArrayList;
import java.util.Map;

import org.apache.commons.math3.distribution.ExponentialDistribution;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.gpu_ai.AILoadGenerator;
import edu.boun.edgecloudsim.gpu_ai.CNNModel;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;

/**
 * Phase 2 load generator — produces {@link AITaskPropertyDual} entries.
 *
 * <p>Extends {@link AILoadGenerator} and overrides {@code initializeModel()}
 * in full, substituting {@code AITaskPropertyDual} for {@code AITaskProperty}.
 * All other behaviour (device assignment, Poisson inter-arrival, active/idle
 * periods) is identical.</p>
 */
public class AILoadGeneratorDual extends AILoadGenerator {

    /** Mirrors AILoadGenerator.taskTypeOfDevices (private in parent). */
    private int[] taskTypeOfDevicesDual;

    public AILoadGeneratorDual(int numOfMobileDevices, double simulationTime,
                               String simScenario,
                               Map<String, CNNModel> appModelMap,
                               int batchSize, int concurrentStreams,
                               double dutyCyclePct, double minAccuracy) {
        super(numOfMobileDevices, simulationTime, simScenario,
              appModelMap, batchSize, concurrentStreams, dutyCyclePct, minAccuracy);
    }

    /**
     * Override: creates {@link AITaskPropertyDual} instead of {@code AITaskProperty}.
     * Logic is identical to {@code AILoadGenerator.initializeModel()}.
     */
    @Override
    public void initializeModel() {
        taskList = new ArrayList<>();

        SimSettings ss     = SimSettings.getInstance();
        double[][]  lookup = ss.getTaskLookUpTable();
        int numTaskTypes   = lookup.length;

        ExponentialDistribution[][] expRng =
                new ExponentialDistribution[numTaskTypes][3];
        for (int i = 0; i < numTaskTypes; i++) {
            if (lookup[i][0] == 0) {
				continue;
			}
            expRng[i][0] = new ExponentialDistribution(lookup[i][5]);
            expRng[i][1] = new ExponentialDistribution(lookup[i][6]);
            expRng[i][2] = new ExponentialDistribution(lookup[i][7]);
        }

        taskTypeOfDevicesDual = new int[numberOfMobileDevices];

        for (int devId = 0; devId < numberOfMobileDevices; devId++) {

            int    taskType = -1;
            double selector = SimUtils.getRandomDoubleNumber(0, 100);
            double cumPct   = 0;
            for (int j = 0; j < numTaskTypes; j++) {
                cumPct += lookup[j][0];
                if (selector <= cumPct) { taskType = j; break; }
            }
            if (taskType == -1) {
                SimLogger.printLine("AILoadGeneratorDual: no task type for device " + devId);
                continue;
            }
            taskTypeOfDevicesDual[devId] = taskType;

            String   appName = ss.getTaskName(taskType);
            CNNModel model   = getAppModelMap().getOrDefault(appName, CNNModel.RESNET50_FP32);
            if (!getAppModelMap().containsKey(appName)) {
                SimLogger.printLine("AILoadGeneratorDual: no CNNModel for '" + appName
                        + "', using ResNet-50 FP32.");
            }

            double poissonMean   = lookup[taskType][2];
            double activePeriod  = lookup[taskType][3];
            double idlePeriod    = lookup[taskType][4];
            int    pesNumber     = (int) lookup[taskType][8];
            double slaLatencyMs  = lookup[taskType][13] * 1000.0;
            if (slaLatencyMs <= 0) {
				slaLatencyMs = 100.0;
			}

            double requestRateHz = (poissonMean > 0) ? (1.0 / poissonMean) : 30.0;
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

                // ── Phase 2: AITaskPropertyDual ──────────────────────────
                // cloudletLength = CPU MI (preprocessing)
                // gpuMFLOP stored separately → will become task.tensors
                taskList.add(new AITaskPropertyDual(
                        virtualTime, devId, taskType, pesNumber,
                        inputSize, outputSize,
                        model,
                        getBatchSize(), getConcurrentStreams(), requestRateHz,
                        preprocessMs, postprocessMs,
                        getDutyCyclePct(),
                        slaLatencyMs, getMinAccuracy()
                ));
            }
        }
    }

    @Override
    public int getTaskTypeOfDevice(int deviceId) {
        if (taskTypeOfDevicesDual == null || deviceId >= taskTypeOfDevicesDual.length) {
			return 0;
		}
        return taskTypeOfDevicesDual[deviceId];
    }
}
