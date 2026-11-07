/*
 * Title:        EdgeCloudSim - AITaskPropertyDual  (Phase 2)
 *
 * Description:
 * Phase 2 version of AITaskProperty.
 * CRITICAL difference: cloudletLength = CPU MI (preprocessing only),
 * NOT GPU MFLOP.  GPU MFLOP is stored in gpuMFLOP field separately.
 *
 * Phase 1 AITaskProperty:  length = computeGpuMFLOP()  (single-channel)
 * Phase 2 AITaskPropertyDual: length = computeCpuMI()  (CPU channel only)
 *                              gpuMFLOP = computeGpuMFLOP() (GPU channel)
 *
 * DualChannelScheduler reads task.tensors (GPU MFLOP) and cl.cloudletLength
 * (CPU MI) to compute max(cpuTime, gpuTime) at runtime.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.gpu_dual;

import edu.boun.edgecloudsim.gpu_ai.AITaskProperty;
import edu.boun.edgecloudsim.gpu_ai.CNNModel;
import edu.boun.edgecloudsim.utils.TaskProperty;

/**
 * Phase 2 task descriptor: splits cloudletLength (CPU MI) from GPU MFLOP.
 *
 * <p>Extends {@link TaskProperty} directly (not {@link AITaskProperty})
 * because {@code AITaskProperty} hard-codes
 * {@code length = computeGpuMFLOP()} in its {@code super()} call and Java
 * does not allow overriding what is passed to a parent constructor.</p>
 *
 * <p><b>Channel values for ResNet-50 batch=1 on TX2:</b>
 * <pre>
 *   length   (CPU MI)    = preprocessMs × 3 = 8 ms × 3,000 MIPS = 24 MI
 *   gpuMFLOP (GPU MFLOP) = computeGpuMFLOP() = 10,178 MFLOP
 *
 *   DualChannelScheduler:
 *     cpuTime = 24     / 3,000     =  8 ms
 *     gpuTime = 10,178 / 135,000   = 75 ms
 *     effectiveTime = max(8, 75)   = 75 ms  ✓
 * </pre>
 * </p>
 */
public class AITaskPropertyDual extends TaskProperty {

    // CNN model and workload params (mirrors AITaskProperty)
    private final CNNModel cnnModel;
    private final int      batchSize;
    private final int      concurrentStreams;
    private final double   requestRateHz;
    private final double   preprocessTimeMs;
    private final double   postprocessTimeMs;
    private final double   dutyCyclePct;
    private final double   slaLatencyMs;
    private final double   minAccuracy;

    /**
     * GPU MFLOP computed from CNNModel — stored separately from cloudletLength.
     * Passed to AITask via the Phase 2 constructor and then to task.tensors.
     */
    private final long gpuMFLOP;

    /**
     * Full constructor.  {@code length} is set to CPU MI only.
     *
     * @param startTime       simulation submit time (s)
     * @param mobileDeviceId  source device id
     * @param taskType        application type index
     * @param pesNumber       CPU cores needed
     * @param inputFileSize   upload bytes
     * @param outputFileSize  download bytes
     * @param cnnModel        CNN model descriptor
     * @param batchSize       GPU batch size
     * @param concurrentStreams parallel streams
     * @param requestRateHz   inferences/second per stream
     * @param preprocessMs    CPU preprocessing time in ms (drives cloudletLength)
     * @param postprocessMs   CPU post-processing time in ms
     * @param dutyCyclePct    GPU duty cycle ceiling (%)
     * @param slaLatencyMs    SLA max latency (ms)
     * @param minAccuracy     SLA min accuracy (%)
     */
    public AITaskPropertyDual(double startTime, int mobileDeviceId, int taskType,
                               int pesNumber, long inputFileSize, long outputFileSize,
                               CNNModel cnnModel,
                               int batchSize, int concurrentStreams,
                               double requestRateHz,
                               double preprocessMs, double postprocessMs,
                               double dutyCyclePct,
                               double slaLatencyMs, double minAccuracy) {

        // length = CPU MI only  ← KEY DIFFERENCE from AITaskProperty (Phase 1)
        super(startTime, mobileDeviceId, taskType, pesNumber,
              AITaskProperty.computeCpuMI(preprocessMs),   // CPU preprocessing MI
              inputFileSize, outputFileSize);

        this.cnnModel          = cnnModel;
        this.batchSize         = batchSize;
        this.concurrentStreams = concurrentStreams;
        this.requestRateHz     = requestRateHz;
        this.preprocessTimeMs  = preprocessMs;
        this.postprocessTimeMs = postprocessMs;
        this.dutyCyclePct      = dutyCyclePct;
        this.slaLatencyMs      = slaLatencyMs;
        this.minAccuracy       = minAccuracy;

        // GPU MFLOP computed once and stored — passed to AITask.tensors at task creation
        this.gpuMFLOP = AITaskProperty.computeGpuMFLOP(cnnModel, batchSize);
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------

    /** GPU MFLOP for this task — must be passed to AITask.setTensors(). */
    public long   getGpuMFLOP()           { return gpuMFLOP; }

    public CNNModel getCnnModel()          { return cnnModel; }
    public int    getBatchSize()           { return batchSize; }
    public int    getConcurrentStreams()   { return concurrentStreams; }
    public double getRequestRateHz()       { return requestRateHz; }
    public double getPreprocessTimeMs()    { return preprocessTimeMs; }
    public double getPostprocessTimeMs()   { return postprocessTimeMs; }
    public double getDutyCyclePct()        { return dutyCyclePct; }
    public double getSlaLatencyMs()        { return slaLatencyMs; }
    public double getMinAccuracy()         { return minAccuracy; }
}
