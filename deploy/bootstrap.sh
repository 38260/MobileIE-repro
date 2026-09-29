#!/bin/sh
# 平板端一键脚本：离线装 LiteRT + 下载模型 + 测速 + 把结果回传给 PC。
# 用法（在 proot-distro 的 Ubuntu 里，默认 root）：
#   curl -s http://<PC_IP>:8770/deploy/bootstrap.sh | sh -s -- http://<PC_IP>:8770
# 前提：PC 上已运行  python deploy/serve.py
set -e
PC="${1:?用法: sh bootstrap.sh http://PC_IP:PORT}"
W=/root/mobileie-bench
mkdir -p "$W"; cd "$W"

echo "[1/5] 系统依赖（apt，需要平板能上网；若平板无网请改用 adb push）"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv wget unzip >/dev/null

echo "[2/5] 取离线 wheel / 测速脚本 / 模型"
wget -q -O wheels.zip "$PC/deploy/wheels.zip"
wget -q -O bench_tflite.py "$PC/deploy/bench_tflite.py"
wget -q -O lolv1.tflite "$PC/MobileIE-main/MobileIE-main/pretrain/lolv1.tflite"
unzip -o -q wheels.zip -d wheels

echo "[3/5] 装 LiteRT（完全离线，从本地 wheel 装，避开 Termux/pip 联网失败）"
python3 -m venv venv
./venv/bin/pip install -q --no-index --find-links wheels ai-edge-litert numpy pillow

echo "[4/5] 测速：4 线程与单线程各 100 帧（warmup 20）"
for t in 4 1; do
  ./venv/bin/python bench_tflite.py --model lolv1.tflite --threads "$t" --iters 100 \
      --json "tablet_t$t.json"
  wget -q -O /dev/null --post-file="tablet_t$t.json" "$PC/upload/tablet_cpu_t$t.json" \
      || echo "  (回传失败；结果留在 $W/tablet_t$t.json)"
done

echo "[5/5] 画质闸门：跑 15 张 eval15，核对 PSNR 是否等于 PC 上的 23.6321 dB"
if wget -q -O list.txt "$PC/deploy/eval15.txt"; then
  mkdir -p inp gt
  while read -r name; do
    [ -n "$name" ] || continue
    wget -q -O "inp/$name" "$PC/repro/data/LOLv1/eval15/low/$name"
    wget -q -O "gt/$name"  "$PC/repro/data/LOLv1/eval15/high/$name"
  done < list.txt
  ./venv/bin/python bench_tflite.py --model lolv1.tflite --threads 4 --iters 20 \
      --quality --inp inp --gt gt --json tablet_quality.json
  wget -q -O /dev/null --post-file=tablet_quality.json "$PC/upload/tablet_quality.json" || true
else
  echo "  跳过画质闸门（PC 上没有 deploy/eval15.txt）"
fi

echo "完成。工作目录：$W"
