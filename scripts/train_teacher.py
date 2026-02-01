"""
教师模型训练脚本
用法: python scripts/train_teacher.py --epochs 100

使用更强的数据增强策略训练增强版教师模型
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def train_teacher(args):
    """训练增强版教师模型"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    exp_name = f"{timestamp}_teacher_enhanced"
    exp_dir = ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"增强版教师模型训练")
    print(f"{'='*60}")
    print(f"实验目录: {exp_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"{'='*60}\n")
    
    # 加载模型配置
    model_cfg = str(ROOT / "models/yolo11m_enhanced.yaml")
    base_weights = "yolo11m-seg.pt"
    
    model = YOLO(model_cfg).load(base_weights)
    
    # 训练（使用更强的数据增强）
    results = model.train(
        data=str(ROOT / "configs/data.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=0,
        project=str(exp_dir),
        name="train",
        workers=8,
        cache=True,  # 缓存数据到内存提升GPU利用率
        exist_ok=True,
        patience=20,
        verbose=True,
        # === 增强数据增强策略 ===
        hsv_h=0.015,        # 色调
        hsv_s=0.7,          # 饱和度
        hsv_v=0.4,          # 亮度
        degrees=10.0,       # 旋转
        translate=0.1,      # 平移
        scale=0.5,          # 缩放
        shear=2.0,          # 剪切
        perspective=0.0,    # 透视
        flipud=0.5,         # 垂直翻转
        fliplr=0.5,         # 水平翻转
        mosaic=1.0,         # Mosaic增强
        mixup=0.1,          # MixUp增强
        copy_paste=0.1,     # Copy-Paste增强
    )
    
    # 保存最佳权重路径
    best_weights = exp_dir / "train/weights/best.pt"
    
    # 提取指标
    rd = results.results_dict
    metrics = {
        "model": "teacher_enhanced",
        "epochs": args.epochs,
        "timestamp": timestamp,
        "box_mAP50": round(float(rd.get("metrics/mAP50(B)", 0)), 4),
        "box_mAP50-95": round(float(rd.get("metrics/mAP50-95(B)", 0)), 4),
        "precision": round(float(rd.get("metrics/precision(B)", 0)), 4),
        "recall": round(float(rd.get("metrics/recall(B)", 0)), 4),
        "mask_mAP50": round(float(rd.get("metrics/mAP50(M)", 0)), 4),
        "mask_mAP50-95": round(float(rd.get("metrics/mAP50-95(M)", 0)), 4),
        "best_weights": str(best_weights),
    }
    
    # 保存指标
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # 创建软链接方便后续使用
    teacher_link = ROOT / "weights" / "teacher_best.pt"
    teacher_link.parent.mkdir(parents=True, exist_ok=True)
    if teacher_link.exists():
        teacher_link.unlink()
    
    # 复制权重到固定位置
    import shutil
    if best_weights.exists():
        shutil.copy(best_weights, teacher_link)
        print(f"\n教师权重已保存到: {teacher_link}")
    
    print(f"\n{'='*60}")
    print(f"教师模型训练完成!")
    print(f"{'='*60}")
    print(f"检测指标:")
    print(f"  mAP50:     {metrics['box_mAP50']:.4f}")
    print(f"  mAP50-95:  {metrics['box_mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"\n最佳权重: {best_weights}")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增强版教师模型训练")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=32,
                        help="批次大小")
    args = parser.parse_args()
    
    train_teacher(args)
