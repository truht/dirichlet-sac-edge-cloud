/*
 * Title:        EdgeCloudSim - AIWorkloadCalculator
 *
 * Description:
 * Implements all 6 computation steps for AI/CNN workload estimation
 * on edge GPU devices, as defined in the research document.
 *
 * Steps:
 *   Step 1: Base CNN inference time from FLOPs and device TFLOPS
 *   Step 2: Architecture correction (RESIDUAL/DENSE/INCEPTION/MOBILE factors)
 *   Step 3: Quantization correction (FP32/FP16/INT8 speed factors)
 *   Step 4: Batch scaling (empirical 0.25 amortisation factor)
 *   Step 5: Add pre/post-process time
 *   Step 6: Thermal throttle correction (duty cycle > 85%)
 *
 * Plus: RAM calculation (5 steps) and GPU load calculation (4 steps).
 *
 * The final result drives the offload decision:
 *   - LATENCY_FINAL > slaLatencyMs  → OFFLOAD
 *   - GPU_LOAD > 90%                → OFFLOAD or reduce batch
 *   - RAM_FINAL > device.freeRAM    → OFFLOAD (OOM risk)
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.gpu_ai;

/**
 * Stateless utility class for AI/CNN workload estimation.
 *
 * <p>All methods are static and side-effect free. The results are packaged
 * into an {@link AIWorkloadResult} record for use by the orchestrator.</p>
 */
public class AIWorkloadCalculator {

    // DL framework RAM overhead constant (from research doc, empirical TensorRT)
    private static final double FRAMEWORK_RAM_BYTES = 400e6; // 400 MB TensorRT

    /** Prevent instantiation of utility class. */
    private AIWorkloadCalculator() {}

    // =======================================================================
    // Main entry point
    // =======================================================================

    /**
     * Computes full workload estimate for a CNN task on a given device.
     *
     * @param model  CNN model descriptor (FLOPs, params, architecture)
     * @param device Target edge GPU device specs
     * @param batch  Batch size for this inference request
     * @param concurrentStreams Number of parallel camera/sensor streams
     * @param requestRateHz    Inference requests per second per stream
     * @param preprocessMs     CPU pre-processing time (ms)
     * @param postprocessMs    CPU post-processing time (ms)
     * @param dutyCyclePct     GPU duty cycle ceiling (0-100)
     * @return AIWorkloadResult with all estimated metrics
     */
    public static AIWorkloadResult calculate(CNNModel model,
                                              EdgeGPUDevice device,
                                              int batch,
                                              int concurrentStreams,
                                              double requestRateHz,
                                              double preprocessMs,
                                              double postprocessMs,
                                              double dutyCyclePct) {
        // --- Latency pipeline ---
        double cnnBaseMs      = step1_baseCNNInference(model, device);
        double cnnAdjusted    = step2_architectureCorrection(cnnBaseMs, model);
        double cnnQuant       = step3_quantizationCorrection(cnnAdjusted, model);
        double cnnBatch       = step4_batchScaling(cnnQuant, batch);
        double totalLatencyMs = step5_addProcessing(cnnBatch, preprocessMs, postprocessMs);
        double latencyFinal   = step6_thermalCorrection(totalLatencyMs, device, dutyCyclePct);

        // --- RAM pipeline ---
        double weightRAM      = calcWeightRAM(model);
        double activationRAM  = calcActivationRAM(model, batch);
        double denseOverhead  = calcDenseOverhead(model, activationRAM);
        double peakRAM_1s     = weightRAM + activationRAM + denseOverhead + FRAMEWORK_RAM_BYTES;
        double peakRAM_total  = calcMultiStreamRAM(weightRAM, activationRAM, denseOverhead, concurrentStreams);
        double ramGB          = peakRAM_total / 1e9;

        // --- GPU load pipeline ---
        double requiredThroughput = calcRequiredThroughput(requestRateHz, concurrentStreams);
        double gpuMaxThroughput   = calcGpuMaxThroughput(cnnBatch);
        double gpuLoadPct         = calcGpuLoad(requiredThroughput, cnnBatch, dutyCyclePct);
        double effectiveFLOPSps   = calcEffectiveFLOPSps(model, batch, requiredThroughput);
        double flopsUtilPct       = Math.min(100.0, effectiveFLOPSps / device.getEffectiveTFLOPS() * 100.0);
        double finalGpuLoad       = Math.min(gpuLoadPct, flopsUtilPct);

        return new AIWorkloadResult(
                latencyFinal, ramGB, finalGpuLoad,
                cnnBaseMs, cnnBatch, requiredThroughput, gpuMaxThroughput,
                peakRAM_1s / 1e9
        );
    }

    // =======================================================================
    // Latency — Step 1: Base CNN inference time
    // =======================================================================

    /**
     * Step 1: Estimate base CNN inference time from FLOPs and effective device TFLOPS.
     *
     * <pre>
     * cnnInferenceMs = (effectiveFLOPs / (ratedTFLOPS × η × powerFactor)) × 1000
     *
     * where η = gpuEfficiency (device.getEffectiveTFLOPS() already includes η):
     *   η_TX2  ≈ 0.18  (calibrated: ResNet-50 51ms measured vs 9ms theoretical)
     *   η_Nano ≈ 0.15  (calibrated: ResNet-50 165ms measured vs 29ms theoretical)
     * </pre>
     *
     * <p><b>Why η is needed:</b> Embedded GPUs achieve only 15–20% of theoretical
     * peak TFLOPS for CNN inference, due to:
     * <ul>
     *   <li>Memory bandwidth bottleneck (not compute-bound for most CNNs)</li>
     *   <li>Kernel launch overhead and pipeline stalls</li>
     *   <li>Data movement between CPU and GPU memory</li>
     * </ul>
     * </p>
     *
     * <p><b>References:</b>
     * <ul>
     *   <li>Dong et al. (2020). LETI: Latency Estimation Tool and Investigation
     *       of Neural Networks on Mobile GPU. <i>arXiv:2010.02871</i></li>
     *   <li>Guo et al. (2023). A hardware-independent time estimation method for
     *       inference of CNN layers on GPU. <i>Parallel Computing</i>,
     *       doi:10.1016/j.parco.2023.103019</li>
     *   <li>Li et al. (2024). A Benchmark for ML Inference Latency on Mobile Devices.
     *       <i>EdgeSys 2024</i></li>
     * </ul>
     * </p>
     */
    public static double step1_baseCNNInference(CNNModel model, EdgeGPUDevice device) {
        // getEffectiveTFLOPS() = ratedTFLOPS × powerModeFactor × gpuEfficiency(η)
        double tflops = device.getEffectiveTFLOPS();
        if (tflops <= 0) {
			return Double.MAX_VALUE;
		}
        return (model.getEffectiveFLOPs() / tflops) * 1000.0;
    }

    // =======================================================================
    // Latency — Step 2: Architecture correction
    // =======================================================================

    /**
     * Step 2: Apply architecture-specific latency correction factor.
     *
     * <pre>
     * archFactor: SEQUENTIAL=1.00, RESIDUAL=0.88, DENSE=1.28,
     *             INCEPTION=0.75, MOBILE=0.92
     * parallelFactor: for INCEPTION with branches, latency = max(branch)/sum
     * </pre>
     *
     * <p><b>Rationale for factors:</b>
     * <ul>
     *   <li>RESIDUAL (0.88): Skip connections improve GPU pipeline reuse.
     *       Reference: He et al. (2016). Deep Residual Learning. <i>CVPR 2016</i></li>
     *   <li>DENSE (1.28): Feature reuse requires re-reading activations, ~20%
     *       overhead. Reference: Pleiss et al. (2017). Memory-Efficient DenseNets.
     *       <i>arXiv:1707.06990</i></li>
     *   <li>INCEPTION (0.75): Parallel branches — latency = max(branch) not sum.
     *       Reference: Szegedy et al. (2015). GoogLeNet. <i>CVPR 2015</i></li>
     *   <li>MOBILE (0.92): Depthwise separable conv reduces FLOPs but adds
     *       memory access overhead. Reference: Howard et al. (2017). MobileNets.
     *       <i>arXiv:1704.04861</i></li>
     * </ul>
     * </p>
     */
    public static double step2_architectureCorrection(double cnnBaseMs, CNNModel model) {
        double archFactor = model.getArchType().getLatencyFactor();

        // Parallel branch correction (Inception-style)
        double parallelFactor = 1.0;
        int branches = model.getNumParallelBranches();
        if (branches > 1) {
            // Research doc: GoogleNet 4 branches → factor ≈ 0.35
            // General formula: 1/branches * correction for the slowest branch
            parallelFactor = 1.0 / branches * 1.4; // 1.4 empirical: slowest branch ~40% above average
            parallelFactor = Math.max(0.25, parallelFactor); // floor at 0.25
        }

        return cnnBaseMs * archFactor * parallelFactor;
    }

    // =======================================================================
    // Latency — Step 3: Quantization correction
    // =======================================================================

    /**
     * Step 3: Scale latency by quantization precision factor.
     *
     * <pre>
     * FP32=1.0, FP16=0.5, INT8=0.25
     * Example: ResNet-50 FP32→INT8: 45ms × 0.25 = 11.25ms
     * </pre>
     *
     * <p><b>References:</b>
     * <ul>
     *   <li>NVIDIA (2019). Low Precision Inference on GPU (GTC S9659).
     *       INT8 provides 4× bandwidth reduction vs FP32.</li>
     *   <li>NVIDIA TensorRT (2023). Achieving FP32 Accuracy for INT8 Inference.
     *       https://developer.nvidia.com/blog/achieving-fp32-accuracy-int8-tensorrt/</li>
     *   <li>Wu et al. (2020). Integer Quantization for Deep Learning Inference.
     *       <i>arXiv:2004.09602</i></li>
     * </ul>
     * </p>
     */
    public static double step3_quantizationCorrection(double cnnAdjustedMs, CNNModel model) {
        return cnnAdjustedMs * model.getQuantMode().getLatencyFactor();
    }

    // =======================================================================
    // Latency — Step 4: Batch scaling
    // =======================================================================

    /**
     * Step 4: Apply batch size scaling with GPU pipeline amortisation.
     *
     * <pre>
     * cnnBatch = cnnQuant × (1 + (batchSize - 1) × 0.25)
     * batch=1 → ×1.0 | batch=4 → ×1.75 | batch=8 → ×2.75
     * </pre>
     *
     * <p>The 0.25 amortisation factor reflects GPU pipeline parallelism: each
     * additional image in a batch adds only ~25% extra latency (not 100%)
     * because kernel launch overhead and memory access latency are shared.
     * This is a linear approximation valid for small batches (≤8) on embedded GPUs.
     * </p>
     *
     * <p><b>References:</b>
     * <ul>
     *   <li>Cui et al. (2024). Characterizing and understanding DNN batching
     *       systems on GPUs. <i>High-Performance Computing</i>,
     *       doi:10.1016/j.hpdc.2024.100012</li>
     *   <li>Zhang et al. (2023). Throughput Maximization of DNN Inference:
     *       Batching or Multi-Tenancy? <i>arXiv:2308.13803</i></li>
     *   <li>NVIDIA Developer Blog (2016). Inference: The Next Step in
     *       GPU-Accelerated Deep Learning.</li>
     * </ul>
     * </p>
     */
    public static double step4_batchScaling(double cnnQuantMs, int batchSize) {
        double scalingFactor = 1.0 + (batchSize - 1) * 0.25;
        return cnnQuantMs * scalingFactor;
    }

    // =======================================================================
    // Latency — Step 5: Add preprocessing + postprocessing
    // =======================================================================

    /**
     * Step 5: Add CPU pre/post-processing time to get end-to-end latency.
     *
     * <pre>
     * totalLatency = preprocessMs + cnnBatch + postprocessMs
     * Example YOLO TX2: 8 + 38 + 15 = 61ms
     * </pre>
     */
    public static double step5_addProcessing(double cnnBatchMs,
                                              double preprocessMs,
                                              double postprocessMs) {
        return preprocessMs + cnnBatchMs + postprocessMs;
    }

    // =======================================================================
    // Latency — Step 6: Thermal throttle correction
    // =======================================================================

    /**
     * Step 6: Apply thermal throttle correction factor.
     *
     * <pre>
     * thermalFactor = dutyCycle > 85% ? 1.0 + (dutyCycle-85)/100 × 2.0 : 1.0
     * duty=100% → factor=1.30 (throttle slows down 30%)
     * duty=75%  → factor=1.00 (GPU has cooling headroom)
     * </pre>
     *
     * <p>Thermal throttling on Jetson devices reduces clock frequency dynamically
     * when temperature exceeds threshold (~75°C), causing throughput degradation
     * of up to 19.2% under sustained load.
     * </p>
     *
     * <p><b>References:</b>
     * <ul>
     *   <li>Rodrigues et al. (2020). Impact of Thermal Throttling on Long-Term
     *       Visual Inference in a CPU-Based Edge Device. <i>arXiv:2010.06291</i>.
     *       Reports 19.2% throughput loss after thermal throttle onset.</li>
     *   <li>Hu et al. (2024). Lotus: Learning-based Online Thermal and Latency
     *       Variation Management for Two-Stage Detectors on Edge Devices.
     *       <i>arXiv:2410.10847</i></li>
     *   <li>NVIDIA Developer Blog. Power Optimization with NVIDIA Jetson.</li>
     * </ul>
     * </p>
     */
    public static double step6_thermalCorrection(double totalLatencyMs,
                                                  EdgeGPUDevice device,
                                                  double dutyCyclePct) {
        double thermalFactor = device.getThermalFactor(dutyCyclePct);
        return totalLatencyMs * thermalFactor;
    }

    // =======================================================================
    // RAM — Step 1: Weight storage RAM
    // =======================================================================

    /**
     * RAM Step 1: Memory for model weights (shared across streams).
     *
     * <pre>
     * bytesPerParam = quantMode.bytesPerParam  (FP32=4, FP16=2, INT8=1)
     * weightRAM = (paramsConv + paramsBN + paramsFC) × bytesPerParam
     * Example ResNet-50 FP32: (21.3M+0.1M+2.0M) × 4 = 93.6 MB
     * </pre>
     */
    public static double calcWeightRAM(CNNModel model) {
        double bpp = model.getQuantMode().getBytesPerParam();
        return model.getTotalParams() * bpp;
    }

    // =======================================================================
    // RAM — Step 2: Activation (feature map) RAM
    // =======================================================================

    /**
     * RAM Step 2: Memory for intermediate feature maps (per batch).
     *
     * <pre>
     * activationRAM = activations × 4.0 × batchSize
     * (Activations always kept in FP32 during inference even if weights are INT8)
     * </pre>
     */
    public static double calcActivationRAM(CNNModel model, int batchSize) {
        return model.getActivations() * 4.0 * batchSize;
    }

    // =======================================================================
    // RAM — Step 3: DenseNet re-read overhead
    // =======================================================================

    /**
     * RAM Step 3: Extra buffer overhead for dense connection cache re-reads.
     *
     * <pre>
     * denseOverhead = activationRAM × memoryReuseRatio
     * DenseNet: 44.2M activations × 4 × 0.80 = 141.4 MB overhead
     * </pre>
     */
    public static double calcDenseOverhead(CNNModel model, double activationRAM) {
        return activationRAM * model.getMemoryReuseRatio();
    }

    // =======================================================================
    // RAM — Step 4+5: Multi-stream peak RAM
    // =======================================================================

    /**
     * RAM Step 4+5: Total peak RAM for concurrent streams.
     *
     * <pre>
     * Weights are shared (loaded once), activations multiply per stream.
     * peakRAM = weightRAM                                   ← shared
     *         + (activationRAM + denseOverhead) × min(streams, 4)
     *         + FRAMEWORK_RAM                               ← one instance
     *
     * Cap at min(streams,4): empirically >4 streams don't add more
     * (GPU memory scheduler reuses slots)
     * </pre>
     */
    public static double calcMultiStreamRAM(double weightRAM,
                                             double activationRAM,
                                             double denseOverhead,
                                             int concurrentStreams) {
        int effectiveStreams = Math.min(concurrentStreams, 4);
        return weightRAM
             + (activationRAM + denseOverhead) * effectiveStreams
             + FRAMEWORK_RAM_BYTES;
    }

    // =======================================================================
    // GPU Load — Required throughput
    // =======================================================================

    /**
     * GPU Load Step 1: Total inference throughput required.
     *
     * <pre>
     * requiredThroughput = requestRateHz × concurrentStreams
     * Example: 30fps × 4 cameras = 120 inf/s
     * </pre>
     */
    public static double calcRequiredThroughput(double requestRateHz, int concurrentStreams) {
        return requestRateHz * concurrentStreams;
    }

    // =======================================================================
    // GPU Load — Max GPU throughput
    // =======================================================================

    /**
     * GPU Load Step 2: Maximum GPU throughput capacity.
     *
     * <pre>
     * gpuMaxThroughput = 1000 / cnnBatchMs  (inf/s)
     * ResNet-50 TX2 batch=1: 1000/45 = 22.2 inf/s
     * </pre>
     */
    public static double calcGpuMaxThroughput(double cnnBatchMs) {
        if (cnnBatchMs <= 0) {
			return 0;
		}
        return 1000.0 / cnnBatchMs;
    }

    // =======================================================================
    // GPU Load — GPU utilisation percentage
    // =======================================================================

    /**
     * GPU Load Step 3: Estimated GPU utilisation percentage.
     *
     * <pre>
     * Formula derivation:
     *   time_needed_per_second = requiredThroughput (inf/s) × cnnBatchMs/1000 (s/inf)
     *   time_available_per_second = dutyCycle / 100
     *   gpuLoadPct = (time_needed / time_available) × 100
     *              = (requiredThroughput × cnnBatchMs/1000) / (dutyCycle/100) × 100
     *              = (requiredThroughput × cnnBatchMs) / (10 × dutyCycle)  [simplified]
     *
     * Example: 4 streams × 30fps = 120 inf/s, ResNet-50 TX2 cnnBatch=50.3ms, duty=75%:
     *   = (120 × 50.3) / (10 × 75) = 804.8% → capped 100% → OFFLOAD ✓
     *
     * Example: 1 stream × 10fps = 10 inf/s, MobileNet Nano cnnBatch=7.8ms, duty=75%:
     *   = (10 × 7.8) / (10 × 75) = 1.04% → LOCAL ✓
     * </pre>
     *
     * <p><b>Reference:</b> Zhang et al. (2023). Throughput Maximization of DNN Inference:
     * Batching or Multi-Tenancy? <i>arXiv:2308.13803</i></p>
     *
     * @param requiredThroughput total inf/s needed (requestRateHz × concurrentStreams)
     * @param cnnBatchMs         inference time for the configured batch (ms)
     * @param dutyCyclePct       GPU duty cycle ceiling (0-100)
     */
    public static double calcGpuLoad(double requiredThroughput,
                                      double cnnBatchMs,
                                      double dutyCyclePct) {
        // Required GPU time per second (ms/s):
        //   requiredThroughput (inf/s) × cnnBatchMs (ms/inf) = ms of GPU work per second
        double requiredGpuMs = requiredThroughput * cnnBatchMs;

        // Available GPU time per second (ms/s):
        //   1000 ms × dutyCycle fraction
        double duty = Math.max(1.0, dutyCyclePct) / 100.0;   // convert % → fraction
        double availableGpuMs = 1000.0 * duty;

        // Load fraction: >1.0 means overloaded, convert to percentage
        // Example: ResNet TX2, 4×30fps:
        //   required = 120 × 50.3 = 6036 ms/s
        //   available = 1000 × 0.75 = 750 ms/s
        //   load = 6036/750 × 100 = 804.8% → capped 100% → OFFLOAD
        double load = (requiredGpuMs / availableGpuMs) * 100.0;
        return Math.min(100.0, load);
    }

    // =======================================================================
    // GPU Load — Effective FLOPS/s processed
    // =======================================================================

    /**
     * GPU Load Step 4: Effective FLOPS processed per second.
     *
     * <pre>
     * effectiveFLOPSps = effectiveFLOPs × batchSize × requiredThroughput
     * ResNet-50 4streams 30fps: 7.71G × 1 × 120 = 925.2 GFLOPS/s
     * TX2 max = 1330 GFLOPS → util = 925/1330 = 69.5%
     * </pre>
     */
    public static double calcEffectiveFLOPSps(CNNModel model,
                                               int batchSize,
                                               double requiredThroughput) {
        return model.getEffectiveFLOPs() * batchSize * requiredThroughput;
    }

    // =======================================================================
    // Offload decision helper
    // =======================================================================

    /**
     * Evaluates whether the AI task should be offloaded based on computed metrics.
     *
     * <p>Decision rules (from research document):
     * <ol>
     *   <li>latencyFinal > slaLatencyMs          → OFFLOAD</li>
     *   <li>gpuLoadPct >= 90%                    → OFFLOAD (or reduce batch)</li>
     *   <li>ramGB > device.freeRAMGB             → OFFLOAD (OOM risk)</li>
     * </ol>
     * If all three pass → LOCAL_OK
     * </p>
     *
     * @param result       result from {@link #calculate}
     * @param device       target device (for free RAM check)
     * @param slaLatencyMs SLA latency requirement (ms)
     * @param minAccuracy  minimum acceptable model accuracy (%)
     * @param modelAccuracy actual model accuracy
     * @return {@link OffloadDecision}
     */
    public static OffloadDecision decide(AIWorkloadResult result,
                                          EdgeGPUDevice device,
                                          double slaLatencyMs,
                                          double minAccuracy,
                                          double modelAccuracy) {
        boolean latencyOk  = result.getLatencyFinalMs() <= slaLatencyMs;
        boolean gpuOk      = result.getGpuLoadPct() < 90.0;
        boolean ramOk      = result.getRamGB() <= device.getFreeRAMGB();
        boolean accuracyOk = modelAccuracy >= minAccuracy;

        if (latencyOk && gpuOk && ramOk && accuracyOk) {
            return OffloadDecision.LOCAL_OK;
        }

        StringBuilder reason = new StringBuilder();
        if (!latencyOk) {
			reason.append("LATENCY(").append(String.format("%.1f", result.getLatencyFinalMs())).append("ms>").append(slaLatencyMs).append(") ");
		}
        if (!gpuOk) {
			reason.append("GPU_OVERLOAD(").append(String.format("%.1f", result.getGpuLoadPct())).append("%) ");
		}
        if (!ramOk) {
			reason.append("OOM(").append(String.format("%.2f", result.getRamGB())).append("GB>").append(String.format("%.2f", device.getFreeRAMGB())).append("GB) ");
		}
        if (!accuracyOk) {
			reason.append("ACCURACY(").append(modelAccuracy).append("<").append(minAccuracy).append(") ");
		}

        return new OffloadDecision(true, reason.toString().trim());
    }

    // =======================================================================
    // Inner result classes
    // =======================================================================

    /**
     * Immutable container for all workload calculation results.
     */
    public static class AIWorkloadResult {
        private final double latencyFinalMs;
        private final double ramGB;
        private final double gpuLoadPct;
        private final double cnnBaseMs;
        private final double cnnBatchMs;
        private final double requiredThroughput;
        private final double gpuMaxThroughput;
        private final double peakRAMPerStreamGB;

        public AIWorkloadResult(double latencyFinalMs, double ramGB, double gpuLoadPct,
                                double cnnBaseMs, double cnnBatchMs,
                                double requiredThroughput, double gpuMaxThroughput,
                                double peakRAMPerStreamGB) {
            this.latencyFinalMs      = latencyFinalMs;
            this.ramGB               = ramGB;
            this.gpuLoadPct          = gpuLoadPct;
            this.cnnBaseMs           = cnnBaseMs;
            this.cnnBatchMs          = cnnBatchMs;
            this.requiredThroughput  = requiredThroughput;
            this.gpuMaxThroughput    = gpuMaxThroughput;
            this.peakRAMPerStreamGB  = peakRAMPerStreamGB;
        }

        public double getLatencyFinalMs()     { return latencyFinalMs; }
        public double getRamGB()              { return ramGB; }
        public double getGpuLoadPct()         { return gpuLoadPct; }
        public double getCnnBaseMs()          { return cnnBaseMs; }
        public double getCnnBatchMs()         { return cnnBatchMs; }
        public double getRequiredThroughput() { return requiredThroughput; }
        public double getGpuMaxThroughput()   { return gpuMaxThroughput; }
        public double getPeakRAMPerStreamGB() { return peakRAMPerStreamGB; }

        @Override
        public String toString() {
            return String.format(
                "AIWorkloadResult[latency=%.1fms | gpuLoad=%.1f%% | RAM=%.2fGB | " +
                "cnnBase=%.1fms | cnnBatch=%.1fms | throughput=%.1f/%.1f inf/s]",
                latencyFinalMs, gpuLoadPct, ramGB,
                cnnBaseMs, cnnBatchMs, requiredThroughput, gpuMaxThroughput);
        }
    }

    /**
     * Result of the offload decision evaluation.
     */
    public static class OffloadDecision {
        private final boolean shouldOffload;
        private final String  reason;

        public static final OffloadDecision LOCAL_OK =
                new OffloadDecision(false, "All constraints satisfied — run locally");

        public OffloadDecision(boolean shouldOffload, String reason) {
            this.shouldOffload = shouldOffload;
            this.reason = reason;
        }

        public boolean shouldOffload() { return shouldOffload; }
        public String getReason()      { return reason; }

        @Override
        public String toString() {
            return shouldOffload ? "OFFLOAD [" + reason + "]" : "LOCAL_OK";
        }
    }
}
