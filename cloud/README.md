# cloud/ —— 云端 bs=4 忠实训练交付包

这个目录只干一件事：**把 LOLv1 的两阶段忠实训练搬到一张 16–24 GB 的云上卡上跑**，
好让 `batch_size` 回到官方 `config/lle.yaml` 的值（4），并让这台 12 GB 笔记本腾出来。

```
cloud/
  configs/
    lolv1_cloud_stage1.yaml         stage 1, bs=4, warmup10+1000ep       -> runs/lolv1_cloud_s1
    lolv1_cloud_stage2_iwo.yaml     stage 2 IWO, bs=4, 1000ep            -> runs/lolv1_cloud_s2
    lolv1_cloud_ctrl_noiwo.yaml     对照组：无 IWO，连跑 2000ep（Fig.10） -> runs/lolv1_cloud_ctrl
  pack.ps1        打上传包（默认只带 LOLv1，约 340 MB）
  bootstrap.sh    实例上一键：装环境 -> 自检 -> 定时探针 -> 后台挂两阶段
  sync_back.ps1   笔记本上另开一个终端：每 5 分钟把 runs/ 和 logs/ 拉回本地
  README.md       本文件
```

## 五步流程

```powershell
# ① 本地打包（先 -DryRun 看清单）
powershell -File cloud\pack.ps1 -DryRun
powershell -File cloud\pack.ps1                 # -> cloud\dist\MobileIE-cloud.tar.gz (~340 MB)
```
```bash
# ② 传上去（AutoDL/恒源云等给了 ssh 的都用这个；Kaggle 改成上传成 Dataset）
scp -P <port> cloud/dist/MobileIE-cloud.tar.gz root@<host>:/root/
ssh  -p <port> root@<host>
cd /root && tar xzf MobileIE-cloud.tar.gz

# ③ 一条命令跑完（镜像自带 torch 时 SKIP_TORCH=1 是默认；裸镜像用 SKIP_TORCH=0）
bash MobileIE/cloud/bootstrap.sh

# ④ 立刻回到笔记本，另开终端把产物往外拉（实例被抢占/销毁就靠这条保命）
powershell -File cloud\sync_back.ps1 -User root -Address <host> -Port <port> -RemoteDir /root/MobileIE -IntervalSec 300

# ⑤ 跑完拉回 runs/，本地出终版报告（不占独显）
#    sync_back 落在 cloud/remote/repro/runs 下，先搬回 repro/runs 再评估
robocopy cloud\remote\repro\runs repro\runs /E
cd repro
python scripts/evaluate.py --config ..\cloud\configs\lolv1_cloud_stage1.yaml --run-name lolv1_cloud_s1 --tag cloud_s1
python scripts/evaluate.py --config ..\cloud\configs\lolv1_cloud_stage2_iwo.yaml --run-name lolv1_cloud_s2 --tag cloud_s2
python scripts/report.py          # 报告会自动把 runs/lolv1_cloud_* 一并列进表
```

第 ⑤ 步用 `..\cloud\configs\*.yaml` 这种路径写法，是因为这三份配置住在 `cloud/configs/`；
`pack.ps1` 打包时会把它们同时放进包内的 `repro/configs/`，所以在实例上可以直接按名字用
（`--config lolv1_cloud_stage1`）。

## 时长与花费

**本机实测锚点**（不是估计）：

| 配法 | s/epoch | 1000 epoch | 峰值显存 |
|---|---|---|---|
| bs=2 fp32 全图 | 31.8 s | 8.8 h | 6.00 GB |
| bs=4 fp32 全图（官方值，本机跑过 1 epoch 验证） | **66 s**（含首轮 cuDNN 自动调优；稳态 46 s） | 12.8–18 h | **10.98 GB**（12.8 GB 卡只剩 1.8 GB 余量） |

> 由此得一条反直觉但已实测的结论：**这张 12 GB 笔记本上 bs=4 不但更吃显存，还比 bs=2 每样本慢 57%**
> —— 训练图是多分支 concat 造成的带宽瓶颈，加大 batch 不省钱。bs=4 的意义只在「与官方配置一致」。

**云端时长 = 探针实测，不是估的**：`bootstrap.sh` 会先跑 3 个真实 epoch 打印 `### measured Ns/epoch`
和换算好的 stage1/两阶段/对照时长，再决定挂多久。参考价（2026-07 快照）：4090 ¥2.19/时、
3090 ¥1.66/时、A100 40G ¥3.45/时 —— 标定用的是**同一模型同一分辨率的实测延迟比**（本机每帧同步比
4090 慢 1.35×，流水线口径 1.11×）加上带宽比 ~1.5×：两阶段估 **9–13 h ≈ ¥20–29**，完整三档估
**18–26 h ≈ ¥40–57**（**推算，
以探针实测为准**）。

## 数据集现状（已用 `repro/scripts/verify_dataset.py` 逐项验过）

| 数据集 | 状态 | 说明 |
|---|---|---|
| LOLv1 | ✅ 完整 | 485+485 / 15+15，全 600×400，逐名配对，全部可解码 |
| UIEB | ✅ 完整 | train 800 对 + test 90 对，PNG 620×460，input↔gt 配对通过。**注意官方是 890/60，此镜像为 800/90 重划分**，引用时要标注 |
| ZRR | ✅ 完整（46,839 训练块 + 1,204 测试块） | `train/raw ↔ train/srgb` 各 46,839、`test/*` 各 1,204，448×448 预切块，配对与解码全通过；**Table 3 已核对：PSNR 21.438 vs 论文 21.43、SSIM 0.7312 vs 0.731** |

关于位深的一个坑：ZRR 的 raw 块**容器位深是按内容自动选的** —— 抽样里 35% 是 8-bit `L`、65% 是 16-bit `I;16`，
看起来像污染，但实测 8-bit 那批 max 中位 330、**只有 1% 恰好饱和在 255**（即没有截断），
且 16-bit 那批 0% ≤255，两者完全互补 → 数值是无损的线性 RAW，`÷4095` 对两种容器都成立。
所以 `verify_dataset.py --raw` 的判据已从「max>255」改成「有没有饱和在容器上限」，
前者会因为抽样侥幸而误判（我第一版就被它骗过一次）。

## 断点与安全（这套已经内置，别绕过它）

* 每 epoch 原子写 `runs/<name>/ckpt/state_last.pt`（73 KB：权重 + Adam 矩 + cosine 周期位置 + best + RNG）。
* `run_two_stage.sh` 带自动重试：非零退出 → 等 20 s → `--resume` 原地续跑，最多 4 次，全程写 `logs/runner.log`。
  所以**实例被抢占不用重头再来**，但前提是 `runs/` 还在——这就是第 ④ 步存在的理由。
* 跨机/跨会话续训已实测：loss 从 1.904 平滑接到 1.730，不回弹，`best` 正确继承。
* 换卡**不会逐位复现**：`cudnn.benchmark` 会选不同卷积算法（本机同 seed 两跑 epoch1 loss 就差 1e-4）。
  要和本地数字严格可比，两边都设 `optim.cudnn_benchmark: false`。

## 云端跑完后，报告里剩下的「偏离论文」清单

只剩一条：**单卡 vs 作者那台至少 6 卡的机器**（`main.py` 里硬编码 `CUDA_VISIBLE_DEVICES='5'` 的证据）。
batch、epoch、精度、数据形式、优化器/调度、损失组合全部与官方一致。
</content>
