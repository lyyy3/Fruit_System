#!/bin/bash
# CWD蒸馏实验脚本（仅蒸馏部分）
# 前置条件：教师模型已训练完成
# 用法: bash scripts/run_distill.sh [教师权重路径]

set -e
cd "$(dirname "$0")/.."

# 默认教师模型路径
TEACHER=${1:-"weights/teacher_large.pt"}

# 检查教师模型
if [ ! -f "$TEACHER" ]; then
    echo "错误: 找不到教师模型权重 $TEACHER"
    echo ""
    echo "请先训练教师模型:"
    echo "  python scripts/train_teacher.py --model large --epochs 100"
    echo ""
    echo "或指定已有权重:"
    echo "  bash scripts/run_distill.sh weights/your_teacher.pt"
    exit 1
fi

echo "========================================"
echo "CWD蒸馏实验"
echo "方法: Channel-wise Knowledge Distillation"
echo "教师: $TEACHER"
echo "时间: $(date)"
echo "========================================"

# 实验E: 蒸馏到baseline学生（对照组）
echo ""
echo "[1/2] 蒸馏实验E: YOLO11n baseline + 教师蒸馏..."
python scripts/train_distill.py \
    --student baseline \
    --teacher "$TEACHER" \
    --alpha 0.5 \
    --temperature 3.0 \
    --epochs 100 \
    --batch 32

# 实验F: 蒸馏到ECA学生（实验组）
echo ""
echo "[2/2] 蒸馏实验F: YOLO11n ECA + 教师蒸馏..."
python scripts/train_distill.py \
    --student eca \
    --teacher "$TEACHER" \
    --alpha 0.5 \
    --temperature 3.0 \
    --epochs 100 \
    --batch 32

echo ""
echo "========================================"
echo "蒸馏实验完成!"
echo "时间: $(date)"
echo "========================================"
echo ""
echo "结果说明:"
echo "  - 实验E: baseline学生 + 蒸馏"
echo "  - 实验F: ECA学生 + 蒸馏"
echo "  - 预期: F应该比E效果更好，证明ECA有独立贡献"
