/*
 * Title:        EdgeCloudSim - tutorial3_ai_flop - FlopLoadGenerator
 *
 * Description:
 * Bộ sinh tải Poisson (như cũ) nhưng TÁCH BẠCH CPU và GPU:
 *   - cloudletLength (CPU MI)  = <task_length>  (lấy thẳng từ XML)
 *   - gpuMFLOP       (GPU MFLOP) = <gpu_flops>   (lấy thẳng từ XML)
 *
 * Mỗi thiết bị phát một LUỒNG task theo phân phối Poisson (idle/active),
 * giống IdleActiveLoadGenerator. Trục X = SỐ THIẾT BỊ.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.applications.tutorial3_ai_flop;

import java.util.ArrayList;

import org.apache.commons.math3.distribution.ExponentialDistribution;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;

/**
 * Load generator Poisson tạo {@link FlopTaskProperty} (CPU MI tách GPU MFLOP).
 *
 * <p>Chỉ số cột taskLookUpTable (SimSettings.parseApplicationsXML):
 * <pre>
 *   [0] usage_percentage   [2] poisson_interarrival  [3] active_period  [4] idle_period
 *   [5] data_upload        [6] data_download         [7] task_length(=CPU MI)
 *   [8] required_core      [13] max_delay_requirement [14] gpu_flops(=GPU MFLOP)
 *   [15] gpu_vram_mb(=VRAM MB, 0 nếu XML không có)
 * </pre>
 * </p>
 */
public class FlopLoadGenerator extends LoadGeneratorModel {

    private int[] taskTypeOfDevices;

    public FlopLoadGenerator(int numOfMobileDevices, double simulationTime, String simScenario) {
        super(numOfMobileDevices, simulationTime, simScenario);
    }

    @Override
    public void initializeModel() {
        taskList = new ArrayList<>();

        SimSettings ss   = SimSettings.getInstance();
        double[][] lookup = ss.getTaskLookUpTable();
        int numTaskTypes  = lookup.length;

        // RNG mũ cho input/output/CPU-MI (mean lấy từ XML)
        ExponentialDistribution[][] expRng = new ExponentialDistribution[numTaskTypes][3];
        for (int i = 0; i < numTaskTypes; i++) {
            if (lookup[i][0] == 0) {
                continue;
            }
            expRng[i][0] = new ExponentialDistribution(lookup[i][5]); // data_upload
            expRng[i][1] = new ExponentialDistribution(lookup[i][6]); // data_download
            expRng[i][2] = new ExponentialDistribution(lookup[i][7]); // task_length = CPU MI
        }

        taskTypeOfDevices = new int[numberOfMobileDevices];

        for (int devId = 0; devId < numberOfMobileDevices; devId++) {

            // Chọn loại ứng dụng theo usage_percentage
            int taskType = -1;
            double selector = SimUtils.getRandomDoubleNumber(0, 100);
            double cumPct   = 0;
            for (int j = 0; j < numTaskTypes; j++) {
                cumPct += lookup[j][0];
                if (selector <= cumPct) { taskType = j; break; }
            }
            if (taskType == -1) {
                taskType = 0;
            }
            taskTypeOfDevices[devId] = taskType;

            double poissonMean  = lookup[taskType][2];
            double activePeriod = lookup[taskType][3];
            double idlePeriod   = lookup[taskType][4];
            int    pesNumber    = (int)  Math.max(1.0, lookup[taskType][8]);
            // *** GPU MFLOP đặt THẲNG từ XML (cột gpu_flops), KHÔNG tính toán ***
            long   gpuMFLOP     = (long) Math.max(0.0, lookup[taskType][14]);
            // *** VRAM MB đặt THẲNG từ XML (cột gpu_vram_mb), 0 nếu không có ***
            long   vramMB       = (long) Math.max(0.0, lookup[taskType][15]);

            if (poissonMean <= 0) {
                continue; // app không hợp lệ
            }

            double activeStart = SimUtils.getRandomDoubleNumber(
                    SimSettings.CLIENT_ACTIVITY_START_TIME,
                    SimSettings.CLIENT_ACTIVITY_START_TIME + activePeriod);
            double virtualTime = activeStart;

            ExponentialDistribution rng = new ExponentialDistribution(poissonMean);

            while (virtualTime < simulationTime) {
                double interval = rng.sample();
                if (interval <= 0) {
                    continue;
                }
                virtualTime += interval;

                if (virtualTime > activeStart + activePeriod) {
                    activeStart += activePeriod + idlePeriod;
                    virtualTime  = activeStart;
                    continue;
                }

                long inputSize  = (long) expRng[taskType][0].sample();
                long outputSize = (long) expRng[taskType][1].sample();
                long cpuMI      = (long) Math.max(1.0, expRng[taskType][2].sample()); // CPU MI

                taskList.add(new FlopTaskProperty(
                        virtualTime, devId, taskType, pesNumber,
                        inputSize, outputSize,
                        cpuMI, gpuMFLOP, vramMB));
            }
        }

        SimLogger.printLine("FlopLoadGenerator: tao " + taskList.size()
                + " tac vu (CPU MI = task_length, GPU MFLOP = gpu_flops, lay thang tu XML).");
    }

    @Override
    public int getTaskTypeOfDevice(int deviceId) {
        if (taskTypeOfDevices == null || deviceId >= taskTypeOfDevices.length) {
            return 0;
        }
        return taskTypeOfDevices[deviceId];
    }
}
