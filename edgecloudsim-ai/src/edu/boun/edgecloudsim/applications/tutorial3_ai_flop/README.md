# tutorial3_ai_flop — Kịch bản TẢI AI (FLOP đặt trực tiếp trong XML)

Package **mới**, độc lập, **không sửa** tutorial3 / tutorial3_ai cũ.
Tham khảo cấu trúc của `tutorial3`.

## Ý tưởng kịch bản

- Tải AI: tác vụ suy diễn (inference) — **TÁCH BẠCH CPU và GPU**.
- **CPU và GPU đặt giá trị TRỰC TIẾP trong XML — KHÔNG tính toán** (không CNN factor):
  - `<task_length>` = **CPU MI** (tiền xử lý) → kênh CPU.
  - `<gpu_flops>` = **GPU MFLOP** (inference) → kênh GPU (đặt thẳng).
- `processingTime = max(cpuTime, gpuTime)` do `DualChannelScheduler` tính.
- Chỉ **một loại ứng dụng** (`AI_INFERENCE`).
- Sinh tải Poisson (idle/active), mỗi thiết bị một LUỒNG task. **Trục X = SỐ THIẾT BỊ**.
- **3 giải thuật offloading** (`NETWORK_BASED`, `UTILIZATION_BASED`, `RANDOM`) lấy nguyên từ tutorial3.

## Tách CPU / GPU — đặt giá trị ở đâu

Trong `config/applications.xml`:

```
<task_length>24</task_length>      <!-- CPU MI  (kênh CPU) -->
<gpu_flops>10178</gpu_flops>       <!-- GPU MFLOP (kênh GPU, đặt thẳng) -->
```

`DualChannelScheduler` tính (cpu_mips, gpu_mips lấy từ `edge_devices.xml`):

```
cpuTime        = task_length / cpu_mips      (vd 24    / 3000   =  8 ms  trên TX2)
gpuTime        = gpu_flops   / gpu_mips       (vd 10178 / 135000 = 75 ms  trên TX2)
processingTime = max(cpuTime, gpuTime)        = 75 ms  (GPU là nút thắt)
```

- Đổi tải GPU → sửa `<gpu_flops>` (vd 2029 ~ MobileNetV2, 10796 ~ YOLOv5s). Đổi tải CPU → sửa `<task_length>`.
- **Không sửa code.**

> `gpu_flops` là trường XML **mới**, đọc qua `SimSettings` (thêm vào danh sách
> optional attributes — backward-compatible, app cũ không có thì = 0).
> Giá trị này được nạp vào `Task.tensors`; cloudletLength giữ là CPU MI.

## Thành phần

| Thành phần | Lớp dùng | Ghi chú |
|---|---|---|
| Load generator | `FlopLoadGenerator` (mới) | Poisson; length=CPU MI(task_length), gpuMFLOP=gpu_flops |
| Task property | `FlopTaskProperty` (mới) | mang CPU MI + GPU MFLOP riêng |
| Mobile device manager | `FlopMobileDeviceManager` (mới) | copy tutorial3 + `setTensors(gpu_flops)` |
| Edge server | `gpu_dual.DualChannelEdgeServerManager` | VM.mips=gpu_mips, cpu_mips riêng; max(cpu,gpu) |
| Cloud server | `gpu_dual.DualChannelCloudServerManager` | cloud GPU dual-channel |
| Orchestrator | `tutorial3.SampleEdgeOrchestrator` | tái dùng, **không đổi** |
| Network model | `tutorial3.SampleNetworkModel` | tái dùng, **không đổi** |
| Core | `SimSettings` | **thêm** optional attribute `gpu_flops` (additive) |

## Biên dịch & chạy

```sh
cd scripts/tutorial3_ai_flop
sh compile.sh      # cần JDK (javac) trên máy
sh run.sh          # chạy nhanh 1 iteration
```

Kết quả ghi vào `sim_results/tutorial3_ai_flop/iteN/` với tiền tố
`SIMRESULT_<scenario>_<policy>_<số_thiết_bị>DEVICES_...` (định dạng EdgeCloudSim chuẩn).

## Vẽ đồ thị (trục X = số thiết bị)

```sh
cd scripts/tutorial3_ai_flop/python
pip install -r requirements.txt
python plotAvgServiceTime.py     # hoặc plotAvgFailedTask.py, plotAvgNetworkDelay.py, ...
```

Chỉnh trong `python/config.py`:
- `num_iterations` — đặt bằng số iteration đã chạy (mặc định 10),
- `min_devices/step_devices/max_devices` — khớp `default_config.properties`,
- `sim_scenario` (mặc định `DEFAULT_TX2`), `x_axis_label` (mặc định `Number of Clients`).

## Làm đồ thị MƯỢT hơn

Theo thứ tự ưu tiên (giảm nhiễu thật sự, không chỉ làm đẹp):

1. **Chạy nhiều iteration rồi lấy trung bình** — yếu tố quan trọng nhất.
   Đặt `iterationEnd` trong `MainApp` (mặc định 10) và `num_iterations` trong `config.py` cho khớp.
   Đồ thị tự lấy mean + khoảng tin cậy 95%.
2. **Tăng `simulation_time`** (mặc định 30) → nhiều task/điểm → trung bình ổn định hơn.
3. **Nhiều điểm trên trục X** → giảm `mobile_device_counter_size` (vd 100).
4. **Nội suy spline (chỉ làm mượt thị giác):** bật `smooth_curve: True` trong `config.py`
   (đã bật sẵn) — vẽ đường cong spline qua các điểm, vẫn giữ marker + thanh sai số tại điểm thật.
   Đặt `False` để về đường gấp khúc nối điểm.

MATLAB (`matlab/getConfiguration.m`) đã cập nhật nhãn trục X, đường dẫn, số iteration tương ứng.
