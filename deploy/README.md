# 端侧复现：把 4K 参数的 MobileIE 跑在骁龙 888 平板上

论文只给了结果（Table 1 的 `Latency (SoC, ms) = 6.72`）和一句"计算复杂度在单张 4090 与一台
骁龙 8 Gen 3 手机上测得"（§4.1）。**它没写**：runtime 与版本、是否用 delegate、精度
（fp32/fp16/int8）、延迟那一列的输入形状（表头里 `(600×400)` 只挂在 FPS 列上）、有没有 warmup、
是否锁频。所以"复现 6.72 ms"无法严格成立——能复现的是**同一套我们自己定死并全程公开的测量协议**。

## 0. 统一口径（两台机器都用它）

| 项 | 取值 |
|---|---|
| 运行时 | LiteRT（`ai-edge-litert`，纯 Python 接口，x86_64 与 aarch64 都有 wheel） |
| 模型 | `MobileIE-main/pretrain/lolv1.tflite`，fp32，作者权重的折叠推理图 |
| 输入 | `1×400×600×3` NHWC float32，随机数（与像素内容无关，纯算子耗时） |
| 计时 | warmup 20 次 → 100 次，只包 `invoke()`，报 median / p95 / min |
| 不含 | 图像解码、resize、内存拷贝、显示——这些另测，别混进网络延迟 |

## 1. 你要在平板上做的（约 20 分钟，不需要数据线、不需要 Android Studio）

1. 装 **Termux**：用 F-Droid 或 GitHub release 的 APK（Google Play 那个已停更，会失败）。
2. 进 glibc 环境（关键：manylinux wheel 在 Termux 原生 bionic 下装不上）：
   ```sh
   pkg install proot-distro -y
   proot-distro install ubuntu
   proot-distro login ubuntu
   ```
3. 在 Ubuntu 里装依赖：
   ```sh
   apt update && apt install -y python3-pip python3-pil wget
   pip install --break-system-packages numpy ai-edge-litert
   ```
4. 取脚本与模型。**推荐用局域网，免插线**：在 PC 上执行
   ```
   python -m http.server 8000 -d D:\AIWorkSpace\MobileIE
   ```
   平板浏览器打开 `http://<PC 的局域网 IP>:8000/`，下载这两个文件：
   - `deploy/bench_tflite.py`
   - `MobileIE-main/MobileIE-main/pretrain/lolv1.tflite`
   （查 PC 的 IP：`ipconfig` 看无线网卡的 IPv4。平板和 PC 要连同一个 Wi-Fi。）
5. 跑（先测 4 线程，再测单线程，两个数都要）：
   ```sh
   python3 bench_tflite.py --model lolv1.tflite --iters 100 --threads 4
   python3 bench_tflite.py --model lolv1.tflite --iters 100 --threads 1
   python3 bench_tflite.py --model lolv1.tflite --threads 4 --quality \
       --inp eval15_low --gt eval15_high          # 画质闸门，需要那 15 对图
   ```
6. **测的时候固定环境**（888 会热降频，不控环境数字就没法比）：插电源、开"性能模式"、
   去掉保护壳、静置 5 分钟、关掉其他 App、屏幕亮度调低。把三段 JSON 输出整段发我。

预期：888 的大核弱于桌面 CPU，4 线程 median 我**估计落在 40–120 ms**（估算，不是目标）。
真正要回答的是"能不能进 33 ms（30 fps）"。如果进不去，才需要下一步的 delegate 路线。

## 2. 若 CPU 路径不够快，再做：APK + GPU/NNAPI delegate

需要 Android SDK（Android Studio 或命令行），代价 2–3 小时。届时我写一个最小 Kotlin App：
固定图 → `Interpreter` + `TensorProcessor` → GPU(OpenCL) 或 NNAPI delegate → 输出 JSON。
**先别做**，等第 1 步的数字出来再决定。

## 3. 画质闸门（先于速度）

同一份 fp32 权重，任何设备上 PSNR 都应一致：PC 上 tflite = **23.6321 dB**（`.pkl` 是 23.6343）。
平板上若差过 0.05 dB，说明运行时数值有差异（例如被自动转成 fp16），先查清再谈延迟数字。

## 4. 已完成与待填

| 机器 | 线程 | warmup | median | min | p95 | FPS |
|---|---|---|---|---|---|---|
| 本机 AMD64 Windows | 4 | 20 | **19.93 ms** | 18.38 | 21.94 | 50.2 |
| 本机 AMD64 Windows | 2 | 0 | 36.70 ms | 35.18 | — | 27.3 |
| 本机 AMD64 Windows | 1 | 0 | 72.36 ms | 71.59 | — | 13.8 |
| 平板 骁龙 888 | 4 | 20 | 待测 | | | |
| 平板 骁龙 888 | 1 | 20 | 待测 | | | |
| 论文（骁龙 8 Gen 3，口径未说明） | — | — | 6.72 ms | — | — | — |

三行本机数字都是**阶段一训练同时在跑**时测的（CPU 与训练进程共享），所以它们之间可直接比较，
但比"纯净机器"偏慢；平板测完后，若要严格对比请在本机停训后重测一遍 4 线程。

产物：`deploy/bench_tflite.py`、`deploy/x86_baseline.json`。

## 5. 一条更正记录

早先我给你的"桌面 CPU 126.8 ms/帧"是**单线程、无 warmup、且当时训练正在满负荷跑**下测的；
同一条件下现在重测单线程 72.4 ms、4 线程 19.9 ms。差异主要来自线程数（1→4 约 3.6×）与
当时的 CPU 争用。方向不变（没有 GPU 也能跑），但数字以上表为准，`repro/results/report.md` 已同步更正。
