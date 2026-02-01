#!/bin/bash
# 知识蒸馏一键执行脚本
# 前置条件：教师模型已训练完成，权重在 weights/teacher_best.pt
# 用法: bash scripts/run_distill.sh

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "知识蒸馏实验"
echo "时间: $(date)"
echo "========================================"

# 检查教师模型权重
TEACHER="weights/teacher_best.pt"
if [ ! -f "$TEACHER" ]; then
    echo "错误: 找不到教师模型权重 $TEACHER"
    echo "请先运行: python scripts/train_teacher.py"
    exit 1
fi
echo "教师模型: $TEACHER"

# 实验C: 蒸馏到baseline学生 (无ECA)
echo ""
echo "[1/2] 消融实验C: YOLO11n + 蒸馏 (无ECA)..."
python scripts/train_distill.py \
    --student baseline \
    --teacher "$TEACHER" \
    --alpha 0.5 \
    --temperature 3.0 \
    --epochs 50 \
    --batch 32

# 实验D: 蒸馏到ECA学生 (完整方案)
echo ""
echo "[2/2] 消融实验D: YOLO11n + ECA + 蒸馏 (完整方案)..."
python scripts/train_distill.py \
    --student eca \
    --teacher "$TEACHER" \
    --alpha 0.5 \
    --temperature 3.0 \
    --epochs 50 \
    --batch 32

echo ""
echo "========================================"
echo "蒸馏实验完成!"
echo "时间: $(date)"
echo "========================================"
echo ""
echo "实验结果在 experiments/ 目录"
