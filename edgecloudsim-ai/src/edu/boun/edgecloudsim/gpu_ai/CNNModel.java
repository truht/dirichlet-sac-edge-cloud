/*
 * Title:        EdgeCloudSim - CNNModel
 *
 * Description:
 * Data class encapsulating all CNN model characteristics used for
 * GPU workload estimation on edge devices.
 *
 * Structure follows the 6 feature groups defined in the research document:
 *   Group 1: Computational complexity (FLOPs, params)
 *   Group 2: Memory (activations, weighted neurons)
 *   Group 3: Layer structure counts
 *   Group 4: Extended architecture (skip connections, branches)
 *   Group 5: Model quality (accuracy, quantization, head FLOPs)
 *   Group 6: Workload (batch, streams, rate, pre/post-process, duty cycle)
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.gpu_ai;

/**
 * Immutable descriptor for a CNN model's characteristics.
 *
 * <p>This class serves as the "application profile" for AI tasks, equivalent to
 * the task lookup table entries in standard EdgeCloudSim but extended for GPU workloads.</p>
 *
 * <p>All FLOPs values are in raw floating-point operations (not giga/tera).
 * All param counts are in raw integers. Memory values are in bytes.</p>
 */
public class CNNModel {

    // =======================================================================
    // Enumerations
    // =======================================================================

    /**
     * CNN architecture type — affects GPU scheduling and latency correction factor.
     * Latency correction factors (from research document):
     *   SEQUENTIAL = 1.00 (baseline, e.g. AlexNet, VGG)
     *   RESIDUAL   = 0.88 (skip connections improve pipeline, e.g. ResNet)
     *   DENSE      = 1.28 (cache miss overhead, e.g. DenseNet)
     *   INCEPTION  = 0.75 (parallel branches, latency = max not sum, e.g. GoogLeNet)
     *   MOBILE     = 0.92 (depthwise separable conv, e.g. MobileNet)
     */
    public enum ArchType {
        SEQUENTIAL(1.00),
        RESIDUAL(0.88),
        DENSE(1.28),
        INCEPTION(0.75),
        MOBILE(0.92);

        private final double latencyFactor;
        ArchType(double f) { this.latencyFactor = f; }

        /** Returns the architecture-specific latency correction multiplier. */
        public double getLatencyFactor() { return latencyFactor; }
    }

    /**
     * Quantization precision mode.
     * Speed factors (from research document):
     *   FP32 = 1.00× (baseline, 4 bytes/param)
     *   FP16 = 0.50× speed, 2 bytes/param
     *   INT8 = 0.25× speed, 1 byte/param (~1% accuracy drop)
     */
    public enum QuantMode {
        FP32(1.00, 4.0),
        FP16(0.50, 2.0),
        INT8(0.25, 1.0);

        private final double latencyFactor;
        private final double bytesPerParam;
        QuantMode(double lf, double bpp) { this.latencyFactor = lf; this.bytesPerParam = bpp; }

        /** Returns latency multiplier relative to FP32. */
        public double getLatencyFactor() { return latencyFactor; }

        /** Returns bytes used per model parameter. */
        public double getBytesPerParam() { return bytesPerParam; }
    }

    /**
     * Builder for constructing CNNModel instances with readable code.
     */
    public static class Builder {
        // Required
        private final String name;

        // Group 1: Computation
        private double totalFLOPs;
        private long paramsConv;
        private long paramsBN;
        private long paramsFC;

        // Group 2: Memory
        private long activations;
        private long weightedNeurons;

        // Group 3: Layer counts
        private int numHiddenLayers;
        private int numConvLayers;
        private int numBNLayers;
        private int numFCLayers;
        private int numPoolLayers;
        private int inputImageSize; // H*W*C (e.g. 224*224*3 = 150528)

        // Group 4: Extended architecture
        private ArchType archType = ArchType.SEQUENTIAL;
        private int numSkipConnections;
        private int numParallelBranches;
        private double memoryReuseRatio;

        // Group 5: Quality
        private double top1Accuracy;
        private QuantMode quantMode = QuantMode.FP32;
        private double outputHeadFLOPs;

        // Calibration: framework + memory bandwidth overhead factor.
        // Derived from paper (Bouzidi 2022, Table 5) measured TX2 times vs. FLOPs-only prediction.
        // overheadFactor = measured_ms / theoretical_ms
        //   ResNet-50  : 75ms / 50ms = 1.50 (residual pipeline + BN layers)
        //   MobileNetV2: 15ms / 2ms  = 7.35 (framework overhead dominates tiny model)
        //   YOLOv5s    : 80ms / 56ms = 1.42 (multi-scale head adds memory pressure)
        private double overheadFactor = 1.0;

        public Builder(String name) { this.name = name; }

        public Builder totalFLOPs(double v)          { this.totalFLOPs = v; return this; }
        public Builder overheadFactor(double v)      { this.overheadFactor = v; return this; }
        public Builder paramsConv(long v)             { this.paramsConv = v; return this; }
        public Builder paramsBN(long v)               { this.paramsBN = v; return this; }
        public Builder paramsFC(long v)               { this.paramsFC = v; return this; }
        public Builder activations(long v)            { this.activations = v; return this; }
        public Builder weightedNeurons(long v)        { this.weightedNeurons = v; return this; }
        public Builder numHiddenLayers(int v)         { this.numHiddenLayers = v; return this; }
        public Builder numConvLayers(int v)           { this.numConvLayers = v; return this; }
        public Builder numBNLayers(int v)             { this.numBNLayers = v; return this; }
        public Builder numFCLayers(int v)             { this.numFCLayers = v; return this; }
        public Builder numPoolLayers(int v)           { this.numPoolLayers = v; return this; }
        public Builder inputImageSize(int v)          { this.inputImageSize = v; return this; }
        public Builder archType(ArchType v)           { this.archType = v; return this; }
        public Builder numSkipConnections(int v)      { this.numSkipConnections = v; return this; }
        public Builder numParallelBranches(int v)     { this.numParallelBranches = v; return this; }
        public Builder memoryReuseRatio(double v)     { this.memoryReuseRatio = v; return this; }
        public Builder top1Accuracy(double v)         { this.top1Accuracy = v; return this; }
        public Builder quantMode(QuantMode v)         { this.quantMode = v; return this; }
        public Builder outputHeadFLOPs(double v)      { this.outputHeadFLOPs = v; return this; }

        public CNNModel build() { return new CNNModel(this); }
    }

    // =======================================================================
    // Fields
    // =======================================================================

    private final String name;

    // Group 1: Computation
    private final double totalFLOPs;
    private final long paramsConv;
    private final long paramsBN;
    private final long paramsFC;

    // Group 2: Memory
    private final long activations;
    private final long weightedNeurons;

    // Group 3: Layer structure
    private final int numHiddenLayers;
    private final int numConvLayers;
    private final int numBNLayers;
    private final int numFCLayers;
    private final int numPoolLayers;
    private final int inputImageSize;

    // Group 4: Extended architecture
    private final ArchType archType;
    private final int numSkipConnections;
    private final int numParallelBranches;
    private final double memoryReuseRatio;

    // Group 5: Quality
    private final double top1Accuracy;
    private final QuantMode quantMode;
    private final double outputHeadFLOPs;

    // Calibration factor — accounts for DL framework overhead and memory bandwidth bottleneck.
    // Derived from Bouzidi 2022 measured execution times on Jetson TX2.
    private final double overheadFactor;

    // =======================================================================
    // Constructor (via Builder)
    // =======================================================================

    private CNNModel(Builder b) {
        this.name               = b.name;
        this.totalFLOPs         = b.totalFLOPs;
        this.paramsConv         = b.paramsConv;
        this.paramsBN           = b.paramsBN;
        this.paramsFC           = b.paramsFC;
        this.activations        = b.activations;
        this.weightedNeurons    = b.weightedNeurons;
        this.numHiddenLayers    = b.numHiddenLayers;
        this.numConvLayers      = b.numConvLayers;
        this.numBNLayers        = b.numBNLayers;
        this.numFCLayers        = b.numFCLayers;
        this.numPoolLayers      = b.numPoolLayers;
        this.inputImageSize     = b.inputImageSize;
        this.archType           = b.archType;
        this.numSkipConnections = b.numSkipConnections;
        this.numParallelBranches= b.numParallelBranches;
        this.memoryReuseRatio   = b.memoryReuseRatio;
        this.top1Accuracy       = b.top1Accuracy;
        this.quantMode          = b.quantMode;
        this.outputHeadFLOPs    = b.outputHeadFLOPs;
        this.overheadFactor     = b.overheadFactor;
    }

    // =======================================================================
    // Pre-defined common CNN models (from research document data)
    // =======================================================================

    /**
     * ResNet-50 — 7.71G FLOPs, 25.5M params, RESIDUAL architecture.
     * Calibration (Bouzidi 2022, TX2): measured 75ms, FLOPs-only gives 50ms
     *   → overheadFactor = 75/50 = 1.50 (accounts for BN sync + skip-connection memory)
     */
    public static CNNModel RESNET50_FP32 = new Builder("ResNet-50")
        .totalFLOPs(7.71e9)
        .paramsConv(21_300_000L).paramsBN(100_000L).paramsFC(2_048_000L)
        .activations(38_600_000L).weightedNeurons(25_000_000L)
        .numHiddenLayers(50).numConvLayers(49).numBNLayers(49).numFCLayers(1).numPoolLayers(1)
        .inputImageSize(224 * 224 * 3)
        .archType(ArchType.RESIDUAL).numSkipConnections(16).memoryReuseRatio(0.0)
        .top1Accuracy(76.1).quantMode(QuantMode.FP32).outputHeadFLOPs(0)
        .overheadFactor(1.50)
        .build();

    public static CNNModel RESNET50_INT8 = new Builder("ResNet-50-INT8")
        .totalFLOPs(7.71e9)
        .paramsConv(21_300_000L).paramsBN(100_000L).paramsFC(2_048_000L)
        .activations(38_600_000L).weightedNeurons(25_000_000L)
        .numHiddenLayers(50).numConvLayers(49).numBNLayers(49).numFCLayers(1).numPoolLayers(1)
        .inputImageSize(224 * 224 * 3)
        .archType(ArchType.RESIDUAL).numSkipConnections(16).memoryReuseRatio(0.0)
        .top1Accuracy(75.1).quantMode(QuantMode.INT8).outputHeadFLOPs(0)
        .overheadFactor(1.50)
        .build();

    /**
     * MobileNetV2 — 0.3G FLOPs, MOBILE architecture, suited for Jetson Nano.
     * Calibration (Bouzidi 2022, TX2): measured 15ms, FLOPs-only gives 2ms
     *   → overheadFactor = 7.35 (DL framework overhead dominates tiny model — paper key finding)
     */
    public static CNNModel MOBILENETV2 = new Builder("MobileNetV2")
        .totalFLOPs(0.3e9)
        .paramsConv(2_200_000L).paramsBN(50_000L).paramsFC(1_280_000L)
        .activations(12_000_000L).weightedNeurons(3_400_000L)
        .numHiddenLayers(35).numConvLayers(35).numBNLayers(35).numFCLayers(1).numPoolLayers(1)
        .inputImageSize(224 * 224 * 3)
        .archType(ArchType.MOBILE).numSkipConnections(10).memoryReuseRatio(0.0)
        .top1Accuracy(71.8).quantMode(QuantMode.FP32).outputHeadFLOPs(0)
        .overheadFactor(7.35)
        .build();

    /**
     * YOLOv5s — detection model, 8.6G FLOPs (backbone + multi-scale head), 7.2M params.
     * Calibration (Bouzidi 2022 class, TX2): measured ~80ms, FLOPs-only gives 56ms
     *   → overheadFactor = 1.42 (multi-scale detection head adds memory bandwidth pressure)
     */
    public static CNNModel YOLOV5S = new Builder("YOLOv5s")
        .totalFLOPs(7.2e9)
        .paramsConv(6_500_000L).paramsBN(200_000L).paramsFC(500_000L)
        .activations(22_000_000L).weightedNeurons(7_200_000L)
        .numHiddenLayers(24).numConvLayers(24).numBNLayers(24).numFCLayers(0).numPoolLayers(3)
        .inputImageSize(640 * 640 * 3)
        .archType(ArchType.RESIDUAL).numSkipConnections(8).memoryReuseRatio(0.0)
        .top1Accuracy(56.8).quantMode(QuantMode.FP32).outputHeadFLOPs(7.2e9 * 0.2)
        .overheadFactor(1.42)
        .build();

    // =======================================================================
    // Getters
    // =======================================================================

    public String getName()              { return name; }
    public double getTotalFLOPs()        { return totalFLOPs; }
    public long getParamsConv()          { return paramsConv; }
    public long getParamsBN()            { return paramsBN; }
    public long getParamsFC()            { return paramsFC; }
    public long getTotalParams()         { return paramsConv + paramsBN + paramsFC; }
    public long getActivations()         { return activations; }
    public long getWeightedNeurons()     { return weightedNeurons; }
    public int getNumHiddenLayers()      { return numHiddenLayers; }
    public int getNumConvLayers()        { return numConvLayers; }
    public int getNumBNLayers()          { return numBNLayers; }
    public int getNumFCLayers()          { return numFCLayers; }
    public int getNumPoolLayers()        { return numPoolLayers; }
    public int getInputImageSize()       { return inputImageSize; }
    public ArchType getArchType()        { return archType; }
    public int getNumSkipConnections()   { return numSkipConnections; }
    public int getNumParallelBranches()  { return numParallelBranches; }
    public double getMemoryReuseRatio()  { return memoryReuseRatio; }
    public double getTop1Accuracy()      { return top1Accuracy; }
    public QuantMode getQuantMode()      { return quantMode; }
    public double getOutputHeadFLOPs()   { return outputHeadFLOPs; }
    public double getOverheadFactor()    { return overheadFactor; }

    /** Total FLOPs including the detection/segmentation head (if any). */
    public double getEffectiveFLOPs() {
        return totalFLOPs + outputHeadFLOPs;
    }

    @Override
    public String toString() {
        return String.format("CNNModel[%s arch=%s quant=%s FLOPs=%.2eG params=%dM acc=%.1f%%]",
                name, archType, quantMode, totalFLOPs / 1e9,
                getTotalParams() / 1_000_000, top1Accuracy);
    }
}
