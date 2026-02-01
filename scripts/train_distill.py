"""
知识蒸馏训练入口
用法: 
  基础蒸馏: python scripts/train_distill.py --student eca --teacher weights/teacher_best.pt
  继续蒸馏: python scripts/train_distill.py --student eca --resume experiments/xxx/train/weights/best.pt
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# 确保能找到项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.distill_trainer import DistillSegmentationTrainer
from ultralytics import YOLO

# 学生模型配置
STUDENTS = {
    "baseline": "yolo11n-seg.pt",
    "eca": "models/yolo11n_eca.yaml",
}

# 默认教师模型路径
DEFAULT_TEACHER = "weights/teacher_best.pt"


def train_distill(args):
    """执行蒸馏训练"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # 实验名称标注是否为继续训练
    suffix = "_finetune" if args.resume else ""
    exp_name = f"{timestamp}_distill_{args.student}{suffix}"
    exp_dir = ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"知识蒸馏训练")
    print(f"{'='*60}")
    print(f"实验名称: {exp_name}")
    print(f"学生模型: {args.student}")
    print(f"继续训练: {args.resume or '否（从预训练开始）'}")
    print(f"教师模型: {args.teacher}")
    print(f"蒸馏权重 alpha: {args.alpha}")
    print(f"温度 T: {args.temperature}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}\n")
    
    # 准备学生模型配置
    student_cfg = STUDENTS[args.student]
    if student_cfg.endswith(".pt"):
        model_path = student_cfg
    else:
        model_path = str(ROOT / student_cfg)
    
    # 检查教师模型
    teacher_path = Path(args.teacher)
    if not teacher_path.is_absolute():
        teacher_path = ROOT / args.teacher
    if not teacher_path.exists():
        print(f"警告: 教师模型不存在 {teacher_path}")
        print("将使用普通训练（无蒸馏）")
        teacher_path = None
    else:
        teacher_path = str(teacher_path)
    
    # 创建蒸馏训练器
    overrides = {
        "model": model_path,
        "data": str(ROOT / "configs/data.yaml"),
        "epochs": args.epochs,
        "imgsz": 640,
        "batch": args.batch,
        "device": 0,
        "project": str(exp_dir),
        "name": "train",
        "workers": 8,
        "cache": True,
        "exist_ok": True,
        "patience": 20,
        "verbose": True,
    }
    
    trainer = DistillSegmentationTrainer(
        overrides=overrides,
        teacher_weights=teacher_path,
        alpha=args.alpha,
        temperature=args.temperature,
    )
    
    # 加载学生模型权重
    if args.resume:
        # 从已训练的模型继续蒸馏
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / args.resume
        if resume_path.exists():
            print(f"从已训练模型加载权重: {resume_path}")
            trainer.model = YOLO(str(resume_path)).model
        else:
            print(f"警告: 指定的resume权重不存在 {resume_path}，使用预训练权重")
            trainer.model = YOLO(model_path).load("yolo11n-seg.pt").model
    elif not student_cfg.endswith(".pt"):
        # 从预训练权重开始
        base_weights = "yolo11n-seg.pt"
        trainer.model = YOLO(model_path).load(base_weights).model
    
    # 开始训练
    trainer.train()
    
    # 获取结果
    try:
        import thop
        import torch
        params = sum(p.numel() for p in trainer.model.parameters())
        params_m = params / 1e6
        dummy_input = torch.randn(1, 3, 640, 640).to(next(trainer.model.parameters()).device)
        flops, _ = thop.profile(trainer.model, inputs=(dummy_input,), verbose=False)
        flops_g = flops / 1e9
    except Exception as e:
        print(f"获取效率指标失败: {e}")
        params_m = 0
        flops_g = 0
    
    # 保存指标
    metrics = {
        "model": f"distill_{args.student}",
        "teacher": args.teacher,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "epochs": args.epochs,
        "timestamp": timestamp,
        "box_mAP50": round(float(trainer.metrics.get("metrics/mAP50(B)", 0)), 4),
        "box_mAP50-95": round(float(trainer.metrics.get("metrics/mAP50-95(B)", 0)), 4),
        "precision": round(float(trainer.metrics.get("metrics/precision(B)", 0)), 4),
        "recall": round(float(trainer.metrics.get("metrics/recall(B)", 0)), 4),
        "mask_mAP50": round(float(trainer.metrics.get("metrics/mAP50(M)", 0)), 4),
        "mask_mAP50-95": round(float(trainer.metrics.get("metrics/mAP50-95(M)", 0)), 4),
        "params_M": round(params_m, 2),
        "flops_G": round(flops_g, 1),
    }
    
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"蒸馏训练完成: {exp_dir}")
    print(f"{'='*60}")
    print(f"检测指标:")
    print(f"  mAP50:     {metrics['box_mAP50']:.4f}")
    print(f"  mAP50-95:  {metrics['box_mAP50-95']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="知识蒸馏训练")
    parser.add_argument("--student", type=str, default="eca",
                        choices=list(STUDENTS.keys()),
                        help="学生模型类型: baseline, eca")
    parser.add_argument("--teacher", type=str, default=DEFAULT_TEACHER,
                        help="教师模型权重路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="从已训练的模型继续蒸馏（权重路径）")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="蒸馏损失权重 (0-1)，默认0.3")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="软标签温度，默认4.0")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=32,
                        help="批次大小")
    args = parser.parse_args()
    
    train_distill(args)
