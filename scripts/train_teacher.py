"""
教师模型训练脚本
用法: 
  训练YOLO11l教师: python scripts/train_teacher.py --model large --epochs 100
  训练增强版教师:   python scripts/train_teacher.py --model enhanced --epochs 100
"""
import os
import sys
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

# 教师模型配置
TEACHERS = {
    "large": {
        "config": "yolo11l-seg.pt",       # 原始L模型（不含注意力，更强）
        "weights": "yolo11l-seg.pt",
        "name": "teacher_large",
    },
    "enhanced": {
        "config": "models/yolo11m_enhanced.yaml",  # 增强M模型（含ECA）
        "weights": "yolo11m-seg.pt",
        "name": "teacher_enhanced",
    },
    "base": {
        "config": "yolo11m-seg.pt",       # 原始M模型
        "weights": "yolo11m-seg.pt",
        "name": "teacher_base",
    },
}


def train_teacher(args):
    """训练教师模型"""
    teacher_cfg = TEACHERS[args.model]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    exp_name = f"{timestamp}_{teacher_cfg['name']}"
    exp_dir = ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"教师模型训练")
    print(f"{'='*60}")
    print(f"模型类型: {args.model}")
    print(f"配置: {teacher_cfg['config']}")
    print(f"实验目录: {exp_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"{'='*60}\n")
    
    # 加载模型
    cfg = teacher_cfg['config']
    if not cfg.endswith('.pt'):
        cfg = str(ROOT / cfg)
        model = YOLO(cfg).load(teacher_cfg['weights'])
    else:
        model = YOLO(cfg)
    
    # 训练
    results = model.train(
        data=str(ROOT / "configs/data.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=0,
        project=str(exp_dir),
        name="train",
        workers=8,
        cache=True,
        exist_ok=True,
        patience=30,
        verbose=True,
        # 数据增强（对所有教师模型适用）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    # 保存最佳权重路径
    best_weights = exp_dir / "train/weights/best.pt"
    
    # 提取指标
    rd = results.results_dict
    metrics = {
        "model": teacher_cfg['name'],
        "config": teacher_cfg['config'],
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
    
    # 添加各类别AP
    class_names = ["apple", "banana", "citrus", "grape", "kiwi", "strawberry"]
    per_class_metrics = {}
    try:
        val_results = model.val(data=str(ROOT / "configs/data.yaml"), verbose=False)
        if hasattr(val_results, 'box') and val_results.box.ap50 is not None:
            for i, name in enumerate(class_names):
                if i < len(val_results.box.ap50):
                    per_class_metrics[f"{name}_AP50"] = round(float(val_results.box.ap50[i]), 4)
                    per_class_metrics[f"{name}_AP50-95"] = round(float(val_results.box.ap[i]), 4)
    except Exception as e:
        print(f"获取各类别AP失败: {e}")
    
    metrics["per_class"] = per_class_metrics
    
    # 保存指标
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # 复制权重到固定位置
    weights_dir = ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    teacher_link = weights_dir / f"{teacher_cfg['name']}.pt"
    if teacher_link.exists():
        teacher_link.unlink()
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
    print(f"各类别AP50-95:")
    for name in class_names:
        key = f"{name}_AP50-95"
        if key in per_class_metrics:
            print(f"  {name:12s}: {per_class_metrics[key]:.4f}")
    print(f"\n最佳权重: {best_weights}")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="教师模型训练")
    parser.add_argument("--model", type=str, default="large",
                        choices=list(TEACHERS.keys()),
                        help="教师模型类型: large(L模型), enhanced(增强M), base(原始M)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小")
    args = parser.parse_args()
    
    train_teacher(args)
