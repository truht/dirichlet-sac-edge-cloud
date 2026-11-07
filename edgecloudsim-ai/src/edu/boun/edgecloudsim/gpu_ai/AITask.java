/*
 * Title:        EdgeCloudSim - AITask
 *
 * Description:
 * AITask extends the base Task class to add GPU-specific attributes
 * required for AI/CNN workload simulation on edge devices.
 *
 * Key addition: totalFLOPs (Tensors) — mirrors the role of cloudletLength
 * but measures GPU floating-point operations instead of CPU million instructions.
 *
 * Design principle: zero changes to existing CloudSim/EdgeCloudSim code.
 * All GPU attributes are added here as a superset of Task.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.gpu_ai;

import org.cloudbus.cloudsim.UtilizationModel;

import edu.boun.edgecloudsim.edge_client.Task;

/**
 * AITask extends Task with GPU/CNN-specific attributes.
 *
 * <p>In the original CloudSim model, {@code cloudletLength} (MI) drives CPU execution time.
 * For GPU workloads, {@code totalFLOPs} plays the equivalent role: it represents the
 * total floating-point operations of the CNN inference pass and is used to estimate
 * GPU execution time given a device's TFLOPS rating.</p>
 *
 * <p><b>Tensors field analogy:</b>
 * <pre>
 *   CPU: cloudletLength (MI)  /  MIPS  = cpuTime (seconds)
 *   GPU: totalFLOPs    (FLOP) / TFLOPS = gpuTime (seconds)
 * </pre>
 * </p>
 *
 * <p>Additional fields capture CNN model identity, quantization mode, batch size,
 * and workload characteristics needed by {@link AIWorkloadCalculator}.</p>
 */
public class AITask extends Task {

    // -----------------------------------------------------------------------
    // GPU computational load — analogous to cloudletLength for CPU
    // -----------------------------------------------------------------------

    /**
     * Total floating-point operations (FLOPs) for one CNN inference pass.
     * This is the "tensors" / GPU-length of the task.
     * Unit: FLOP (e.g., ResNet-50 ≈ 7.71 × 10^9 FLOPs)
     */
    private double totalFLOPs;

    // -----------------------------------------------------------------------
    // CNN model — full descriptor (single source of truth)
    // -----------------------------------------------------------------------

    /**
     * Full CNN model descriptor carrying FLOPs, architecture, quantization,
     * accuracy and all parameters needed by AIWorkloadCalculator.
     *
     * This replaces individual fields (cnnModelName, archType, quantizationMode)
     * with one object so AIEdgeOrchestrator always has the complete model info.
     */
    private final CNNModel cnnModel;

    // -----------------------------------------------------------------------
    // Workload / streaming parameters
    // -----------------------------------------------------------------------

    /** Number of images processed in one GPU kernel launch (≥1). */
    private int batchSize;

    /** Number of concurrent camera/sensor streams feeding this task. */
    private int concurrentStreams;

    /** Frames (inferences) per second from a single source stream. */
    private double requestRateHz;

    /** CPU preprocessing time in milliseconds (resize, normalize). */
    private double preprocessTimeMs;

    /** CPU post-processing time in milliseconds (NMS, decode, tracking). */
    private double postprocessTimeMs;

    /**
     * Fraction of time the GPU is allowed to run at full speed [0-100].
     * Values below 100 leave thermal headroom; above ~85 triggers throttle modelling.
     */
    private double dutyCyclePct;

    // -----------------------------------------------------------------------
    // Pre-computed results (filled by AIWorkloadCalculator)
    // -----------------------------------------------------------------------

    /** End-to-end latency estimate (ms) computed before offload decision. */
    private double estimatedLatencyMs;

    /** Peak GPU RAM required (GB). */
    private double estimatedRamGB;

    /** Estimated GPU utilisation percentage [0-100]. */
    private double estimatedGpuLoadPct;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /**
     * Full constructor — mirrors Task's constructor and adds GPU fields.
     *
     * @param mobileDeviceId    originating device id
     * @param cloudletId        unique cloudlet id
     * @param cloudletLength    CPU length in MI (kept for compatibility; set to 1 for pure GPU tasks)
     * @param pesNumber         CPU cores required
     * @param cloudletFileSize  upload payload size (bytes)
     * @param cloudletOutputSize download result size (bytes)
     * @param utilizationModelCpu CPU utilisation model
     * @param utilizationModelRam RAM utilisation model
     * @param utilizationModelBw  BW utilisation model
     * @param cnnModel          full CNN model descriptor (FLOPs, arch, quant, accuracy...)
     * @param batchSize         images per GPU batch
     * @param concurrentStreams parallel input streams
     * @param requestRateHz     inferences/second per stream
     * @param preprocessMs      CPU pre-processing time (ms)
     * @param postprocessMs     CPU post-processing time (ms)
     * @param dutyCyclePct      GPU duty cycle ceiling (%)
     */
    public AITask(int mobileDeviceId, int cloudletId,
                  long cloudletLength, int pesNumber,
                  long cloudletFileSize, long cloudletOutputSize,
                  UtilizationModel utilizationModelCpu,
                  UtilizationModel utilizationModelRam,
                  UtilizationModel utilizationModelBw,
                  CNNModel cnnModel,
                  int batchSize,
                  int concurrentStreams,
                  double requestRateHz,
                  double preprocessMs,
                  double postprocessMs,
                  double dutyCyclePct) {

        super(mobileDeviceId, cloudletId, cloudletLength, pesNumber,
              cloudletFileSize, cloudletOutputSize,
              utilizationModelCpu, utilizationModelRam, utilizationModelBw);

        this.cnnModel          = cnnModel;
        this.totalFLOPs        = cnnModel.getTotalFLOPs();   // derived from model
        this.batchSize         = batchSize;
        this.concurrentStreams = concurrentStreams;
        this.requestRateHz     = requestRateHz;
        this.preprocessTimeMs  = preprocessMs;
        this.postprocessTimeMs = postprocessMs;
        this.dutyCyclePct      = dutyCyclePct;

        // ---------------------------------------------------------------
        // Set GPU workload fields on parent Task (used by GpuVm / GpuCloudletScheduler)
        // ---------------------------------------------------------------

        // tensors = same as cloudletLength (both are GPU MFLOP in the current
        // single-channel approach). Kept separate for Phase-2 dual-channel extension
        // where cloudletLength will become CPU MI and tensors the GPU MFLOP.
        setTensors(cloudletLength);   // cloudletLength is already the GPU MFLOP

        // vramRequired = model weights + activations in MB.
        // Used by GpuVm.canFitVram() for admission control.
        //   weights:     paramCount × bytes/param  (FP32=4B, FP16=2B, INT8=1B)
        //   activations: activation count × 4B (FP32 assumed for feature maps)
        long paramBytes = (long)(cnnModel.getTotalParams()
                                 * cnnModel.getQuantMode().getBytesPerParam());
        long activBytes = cnnModel.getActivations() * 4L;   // FP32 activations
        long vramMB     = (paramBytes + activBytes) / (1024L * 1024L);
        setVramRequired(Math.max(1L, vramMB));
    }

    /**
     * Phase 2 constructor — dual-channel: {@code cloudletLength} = CPU MI,
     * {@code gpuMFLOP} = GPU MFLOP stored explicitly in {@code task.tensors}.
     *
     * <p>Used by {@link edu.boun.edgecloudsim.applications.sample_app_ai.AIMobileDeviceManager}
     * when the incoming property is an
     * {@link edu.boun.edgecloudsim.gpu_dual.AITaskPropertyDual}.</p>
     *
     * <p><b>Difference from Phase 1 constructor:</b>
     * <pre>
     *   Phase 1: cloudletLength = GPU MFLOP  → setTensors(cloudletLength)  [same value]
     *   Phase 2: cloudletLength = CPU MI     → setTensors(gpuMFLOP)        [separate value]
     * </pre>
     * DualChannelScheduler uses both to compute max(cpuTime, gpuTime).</p>
     *
     * @param gpuMFLOP GPU MFLOP from AITaskPropertyDual.getGpuMFLOP()
     *                 — stored in tensors, NOT derived from cloudletLength
     */
    public AITask(int mobileDeviceId, int cloudletId,
                  long cloudletLength,   // CPU MI (Phase 2: preprocessing only)
                  long gpuMFLOP,         // GPU MFLOP (Phase 2: explicit, separate field)
                  int pesNumber,
                  long cloudletFileSize, long cloudletOutputSize,
                  UtilizationModel utilizationModelCpu,
                  UtilizationModel utilizationModelRam,
                  UtilizationModel utilizationModelBw,
                  CNNModel cnnModel,
                  int batchSize,
                  int concurrentStreams,
                  double requestRateHz,
                  double preprocessMs,
                  double postprocessMs,
                  double dutyCyclePct) {

        super(mobileDeviceId, cloudletId, cloudletLength, pesNumber,
              cloudletFileSize, cloudletOutputSize,
              utilizationModelCpu, utilizationModelRam, utilizationModelBw);

        this.cnnModel          = cnnModel;
        this.totalFLOPs        = cnnModel.getTotalFLOPs();
        this.batchSize         = batchSize;
        this.concurrentStreams = concurrentStreams;
        this.requestRateHz     = requestRateHz;
        this.preprocessTimeMs  = preprocessMs;
        this.postprocessTimeMs = postprocessMs;
        this.dutyCyclePct      = dutyCyclePct;

        // Phase 2: tensors = explicit gpuMFLOP (NOT cloudletLength)
        setTensors(gpuMFLOP);

        // vramRequired = model weights + activations in MB
        long paramBytes = (long)(cnnModel.getTotalParams()
                                 * cnnModel.getQuantMode().getBytesPerParam());
        long activBytes = cnnModel.getActivations() * 4L;
        long vramMB     = (paramBytes + activBytes) / (1024L * 1024L);
        setVramRequired(Math.max(1L, vramMB));
    }

    // -----------------------------------------------------------------------
    // Getters — GPU / CNN attributes
    // -----------------------------------------------------------------------

    /**
     * Returns the full CNN model descriptor.
     * AIEdgeOrchestrator uses this to run AIWorkloadCalculator.
     */
    public CNNModel getCnnModel() { return cnnModel; }

    /**
     * Returns the total GPU FLOPs for this task ("tensors" / GPU cloudlet length).
     * Derived from cnnModel.getTotalFLOPs().
     */
    public double getTotalFLOPs() { return totalFLOPs; }

    /** Returns the CNN model name (convenience shortcut). */
    public String getCnnModelName() { return cnnModel.getName(); }

    /** Returns the CNN architecture type (convenience shortcut). */
    public CNNModel.ArchType getArchType() { return cnnModel.getArchType(); }

    /** Returns the quantization precision mode (convenience shortcut). */
    public CNNModel.QuantMode getQuantizationMode() { return cnnModel.getQuantMode(); }

    /** Returns the GPU batch size. */
    public int getBatchSize() { return batchSize; }

    /** Returns the number of concurrent input streams. */
    public int getConcurrentStreams() { return concurrentStreams; }

    /** Returns the inference request rate (fps) per stream. */
    public double getRequestRateHz() { return requestRateHz; }

    /** Returns the CPU preprocessing time in ms. */
    public double getPreprocessTimeMs() { return preprocessTimeMs; }

    /** Returns the CPU post-processing time in ms. */
    public double getPostprocessTimeMs() { return postprocessTimeMs; }

    /** Returns the GPU duty cycle ceiling (%). */
    public double getDutyCyclePct() { return dutyCyclePct; }

    // -----------------------------------------------------------------------
    // Getters / setters for pre-computed estimates
    // -----------------------------------------------------------------------

    /** Returns the estimated end-to-end latency in ms (set by AIWorkloadCalculator). */
    public double getEstimatedLatencyMs() { return estimatedLatencyMs; }

    /** Sets the estimated end-to-end latency (called by AIWorkloadCalculator). */
    public void setEstimatedLatencyMs(double ms) { this.estimatedLatencyMs = ms; }

    /** Returns the estimated peak GPU RAM in GB. */
    public double getEstimatedRamGB() { return estimatedRamGB; }

    /** Sets the estimated peak GPU RAM (GB). */
    public void setEstimatedRamGB(double gb) { this.estimatedRamGB = gb; }

    /** Returns the estimated GPU load percentage. */
    public double getEstimatedGpuLoadPct() { return estimatedGpuLoadPct; }

    /** Sets the estimated GPU load percentage. */
    public void setEstimatedGpuLoadPct(double pct) { this.estimatedGpuLoadPct = pct; }

    // -----------------------------------------------------------------------
    // Convenience
    // -----------------------------------------------------------------------

    /**
     * Human-readable summary of key GPU attributes for logging.
     */
    @Override
    public String toString() {
        return String.format("AITask[id=%d, model=%s, FLOPs=%.2eG, batch=%d, "
                + "streams=%d, quant=%s, latEst=%.1fms, ramEst=%.2fGB, gpuLoad=%.1f%%]",
                getCloudletId(), cnnModel.getName(), totalFLOPs / 1e9,
                batchSize, concurrentStreams, cnnModel.getQuantMode(),
                estimatedLatencyMs, estimatedRamGB, estimatedGpuLoadPct);
    }
}
