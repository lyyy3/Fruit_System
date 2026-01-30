#!/bin/bash
# ============================================================
# 批量训练脚本 - 4090D 服务器专用
# 使用方法: ./scripts/run_all.sh
# ============================================================

set -e  # 遇到错误立即退出

# 配置
EPOCHS=100
BATCH_N=32    # n模型用大batch
BATCH_M=16    # m模型batch稍小
DATA="configs/data.yaml"

echo "============================================================"
echo "开始批量训练实验"
echo "Epochs: $EPOCHS"
echo "开始时间: $(date)"
echo "============================================================"

# 激活虚拟环境 (如果有)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 1. YOLO11n baseline
echo ""
echo "[1/4] 训练 YOLO11n baseline..."
python scripts/train.py --model baseline --epochs $EPOCHS --batch $BATCH_N

# 2. YOLO11n + ECA
echo ""
echo "[2/4] 训练 YOLO11n + ECA..."
python scripts/train.py --model eca --epochs $EPOCHS --batch $BATCH_N

# 3. YOLO11m baseline
echo ""
echo "[3/4] 训练 YOLO11m baseline..."
python scripts/train.py --model teacher --epochs $EPOCHS --batch $BATCH_M

# 4. YOLO11m + ECA
echo ""
echo "[4/4] 训练 YOLO11m + ECA..."
python scripts/train.py --model teacher_eca --epochs $EPOCHS --batch $BATCH_M

# 生成对比报告
echo ""
echo "============================================================"
echo "所有训练完成！生成对比报告..."
echo "============================================================"
python scripts/compare.py

echo ""
echo "完成时间: $(date)"
echo "============================================================"
