/*
 * Title:        EdgeCloudSim - AITaskProperty
 *
 * Description:
 * Extends TaskProperty to carry GPU/CNN-specific task parameters.
 * Used by AILoadGenerator to create AITask instances in the simulation.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.gpu_ai;

import edu.boun.edgecloudsim.utils.TaskProperty;

/**
 * AITaskProperty extends {@link TaskProperty} with all AI workload parameters.
 *
 * <p><b>GPU-equivalent cloudletLength (MFLOP):</b><br>
 * In standard EdgeCloudSim, {@code length} is in Million Instructions (MI) and
 * {@code VM.MIPS} is Million Instructions Per Second. For GPU tasks we use:
 * <pre>
 *   length  = corrected FLOPs in Mega-FLOP (MFLOP)
 *   VM.MIPS = device GPU throughput in MFLOPS (set in edge_devices.xml / config)
 *
 *   processingTime = length (MFLOP) / VM.MIPS (MFLOPS) = seconds  ← vật lý đúng
 * </pre>
 * Corrections applied:
 * <ul>
 *   <li>archFactor  — RESIDUAL=0.88, DENSE=1.28, INCEPTION=0.75, MOBILE=0.92</li>
 *   <li>quantFactor — FP32=1.0, FP16=0.5, INT8=0.25</li>
 *   <li>batchFactor — 1 + (batchSize-1) × 0.25</li>
 * </ul>
 * Example ResNet-50 FP32 batch=1:
 * <pre>
 *   MFLOP = 7710M × 0.88(arch) × 1.0(quant) × 1.0(batch) × 1.50(overhead) = 10,178 MFLOP
 *   TX2 VM.MIPS   = 135,000 MFLOPS → processingTime = 10178/135000 = 0.0754s ≈ 75ms ✓
 *   Cloud VM.MIPS = 3,500,000 MFLOPS → processingTime = 10178/3500000 = 0.0029s ≈ 3ms ✓
 *
 *   overheadFactor calibrated from Bouzidi 2022 (Table 5) measured TX2 times:
 *     ResNet-50  : 75ms measured / 50ms FLOPs-only = 1.50
 *     MobileNetV2: 15ms measured / 2ms  FLOPs-only = 7.35 (framework dominates)
 *     YOLOv5s    : 80ms measured / 56ms FLOPs-only = 1.42
 * </pre>
 * </p>
 */
public class AITaskProperty extends TaskProperty {

    // CNN model defining the GPU workload
    private final CNNModel cnnModel;

    // Workload parameters (per task instantiation)
    private final int    batchSize;
    private final int    concurrentStreams;
    private final double requestRateHz;
    private final double preprocessTimeMs;
    private final double postprocessTimeMs;
    private final double dutyCyclePct;

    // SLA constraint for this task
    private final double slaLatencyMs;
    private final double minAccuracy;

    /**
     * Full constructor.
     *
     * @param startTime        simulation time when task is submitted (s)
     * @param mobileDeviceId   source device id
     * @param taskType         application type index
     * @param pesNumber        CPU cores needed (typically 1 for AI tasks)
     * @param inputFileSize    input image bytes (upload payload)
     * @param outputFileSize   result bytes (download payload)
     * @param cnnModel         CNN model descriptor (carries totalFLOPs)
     * @param batchSize        GPU batch size
     * @param concurrentStreams parallel input streams
     * @param requestRateHz    inference rate per stream (fps)
     * @param preprocessMs     CPU pre-processing time (ms)
     * @param postprocessMs    CPU post-processing time (ms)
     * @param dutyCyclePct     GPU duty cycle ceiling (%)
     * @param slaLatencyMs     maximum acceptable end-to-end latency (ms)
     * @param minAccuracy      minimum acceptable model accuracy (%)
     */
    public AITaskProperty(double startTime, int mobileDeviceId, int taskType,
                          int pesNumber, long inputFileSize, long outputFileSize,
                          CNNModel cnnModel,
                          int batchSize, int concurrentStreams,
                          double requestRateHz,
                          double preprocessMs, double postprocessMs,
                          double dutyCyclePct,
                          double slaLatencyMs, double minAccuracy) {

        // super() phải là câu lệnh đầu tiên trong constructor (Java rule).
        // computeGpuMFLOP() là static method nên có thể gọi trực tiếp trong tham số super().
        // VM.MIPS trong edge_devices.xml = GPU TFLOPS × 10^6 (MFLOPS)
        // → processingTime = MFLOP / MFLOPS = seconds  ✓
        super(startTime, mobileDeviceId, taskType, pesNumber,
              computeGpuMFLOP(cnnModel, batchSize),  // GPU-equivalent cloudlet length (MFLOP)
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
    }

    // =======================================================================
    // Getters
    // =======================================================================

    /**
     * Returns the CNN model descriptor — contains totalFLOPs (the "tensors" / GPU length).
     */
    public CNNModel getCnnModel()             { return cnnModel; }

    /**
     * Convenience: returns totalFLOPs from the CNN model.
     */
    public double getTotalFLOPs()             { return cnnModel.getTotalFLOPs(); }

    public int    getBatchSize()              { return batchSize; }
    public int    getConcurrentStreams()      { return concurrentStreams; }
    public double getRequestRateHz()          { return requestRateHz; }
    public double getPreprocessTimeMs()       { return preprocessTimeMs; }
    public double getPostprocessTimeMs()      { return postprocessTimeMs; }
    public double getDutyCyclePct()           { return dutyCyclePct; }
    public double getSlaLatencyMs()           { return slaLatencyMs; }
    public double getMinAccuracy()            { return minAccuracy; }

    // =======================================================================
    // Static helper
    // =======================================================================

    /**
     * Computes GPU-equivalent cloudletLength in MFLOP.
     *
     * <pre>
     * Formula:
     *   correctedFLOPs = totalFLOPs × archFactor × quantFactor × batchFactor
     *   MFLOP          = correctedFLOPs / 10^6
     *
     * Matching VM.MIPS (set in config/edge_devices.xml):
     *   VM.MIPS = deviceTFLOPS × 10^6  (MFLOPS)
     *
     * Result:
     *   processingTime = MFLOP / MFLOPS = seconds  ✓
     * </pre>
     */
    /**
     * Computes CPU preprocessing length in Million Instructions (MI).
     *
     * <p>Used by Phase 2 ({@code AITaskPropertyDual}) to set {@code cloudletLength}
     * to CPU MI only, keeping GPU MFLOP in a separate {@code tensors} field.</p>
     *
     * <pre>
     *   cpuMI = preprocessMs × ARM_MIPS / 1000
     *   ARM Cortex-A57 on TX2 ≈ 3,000 MIPS
     *   Example: 8 ms × 3,000 MIPS / 1000 = 24 MI
     * </pre>
     *
     * @param preprocessMs CPU preprocessing time in milliseconds
     * @return CPU cloudlet length in MI (minimum 1)
     */
    public static long computeCpuMI(double preprocessMs) {
        // ARM Cortex-A57 on Jetson TX2: ~3,000 MIPS
        final double TX2_CPU_MIPS = 3_000.0;
        return Math.max(1L, Math.round(preprocessMs * TX2_CPU_MIPS / 1000.0));
    }

    public static long computeGpuMFLOP(CNNModel model, int batchSize) {
        double effectiveFLOPs = model.getEffectiveFLOPs();             // totalFLOPs + headFLOPs
        double archFactor     = model.getArchType().getLatencyFactor(); // e.g. RESIDUAL=0.88
        double quantFactor    = model.getQuantMode().getLatencyFactor();// e.g. FP32=1.0
        double batchFactor    = 1.0 + (batchSize - 1) * 0.25;          // GPU pipeline amortisation
        double overheadFactor = model.getOverheadFactor();             // framework + memory BW calibration

        double correctedFLOPs = effectiveFLOPs * archFactor * quantFactor * batchFactor * overheadFactor;
        long   mflop          = Math.max(1L, Math.round(correctedFLOPs / 1e6));
        return mflop;
    }
}
