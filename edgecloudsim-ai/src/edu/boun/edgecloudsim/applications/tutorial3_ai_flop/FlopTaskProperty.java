/*
 * Title:        EdgeCloudSim - tutorial3_ai_flop - FlopTaskProperty
 *
 * Description:
 * TaskProperty TÁCH BẠCH CPU và GPU:
 *   - cloudletLength (length) = CPU MI         (lấy thẳng từ <task_length> trong XML)
 *   - gpuMFLOP                = GPU MFLOP        (lấy thẳng từ <gpu_flops> trong XML)
 *
 * gpuMFLOP sẽ được FlopMobileDeviceManager nạp vào Task.tensors. Khi đó
 * DualChannelScheduler tính:
 *   cpuTime       = cloudletLength / cpuMips     (CPU)
 *   gpuTime       = tensors        / gpuMips     (GPU)
 *   processingTime = max(cpuTime, gpuTime)
 *
 * Cả hai giá trị đều ĐẶT TRỰC TIẾP trong file XML — KHÔNG tính toán.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.applications.tutorial3_ai_flop;

import edu.boun.edgecloudsim.utils.TaskProperty;

/**
 * Task descriptor mang thêm GPU MFLOP riêng (tách khỏi CPU MI).
 */
public class FlopTaskProperty extends TaskProperty {

    /** GPU workload (MFLOP) — đặt thẳng từ <gpu_flops>; sẽ nạp vào Task.tensors. */
    private final long gpuMFLOP;

    /**
     * VRAM cần thiết (MB) — đặt thẳng từ <gpu_vram_mb>; sẽ nạp vào
     * Task.setVramRequired(). Field này được thêm cho DRL Constraint C5
     * (xem SimSettings optionalAttributes) — mặc định 0 nếu XML không có,
     * tương thích ngược hoàn toàn với tutorial3_ai_flop gốc (app đó không
     * dùng VRAM nên luôn nhận 0, không đổi hành vi).
     */
    private final long vramMB;

    /**
     * @param startTime       thời điểm phát task (s)
     * @param mobileDeviceId  id thiết bị nguồn
     * @param taskType        loại ứng dụng
     * @param pesNumber       số core CPU
     * @param inputFileSize   upload (KB)
     * @param outputFileSize  download (KB)
     * @param cpuMI           CPU MI  -> cloudletLength (= task_length)
     * @param gpuMFLOP        GPU MFLOP -> tensors      (= gpu_flops)
     * @param vramMB          VRAM MB -> vramRequired   (= gpu_vram_mb, 0 nếu XML không có)
     */
    public FlopTaskProperty(double startTime, int mobileDeviceId, int taskType, int pesNumber,
                            long inputFileSize, long outputFileSize,
                            long cpuMI, long gpuMFLOP, long vramMB) {
        // length = CPU MI (kênh CPU). GPU đi riêng qua tensors.
        super(startTime, mobileDeviceId, taskType, pesNumber,
              cpuMI, inputFileSize, outputFileSize);
        this.gpuMFLOP = gpuMFLOP;
        this.vramMB = vramMB;
    }

    /** GPU MFLOP — phải được nạp vào Task.setTensors() khi tạo Task. */
    public long getGpuMFLOP() { return gpuMFLOP; }

    /** VRAM MB — phải được nạp vào Task.setVramRequired() khi tạo Task. */
    public long getVramMB() { return vramMB; }
}
