"""
统一训练入口
用法: python scripts/train.py --model baseline --epochs 50
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

from ultralytics import YOLO

# 模型配置映射
MODELS = {
    "baseline": "yolo11n-seg.pt",           # 学生baseline
    "eca": "models/yolo11n_eca.yaml",       # 学生+ECA
    "teacher": "yolo11m-seg.pt",            # 教师baseline
    "teacher_eca": "models/yolo11m_eca.yaml", # 教师+ECA
}


def train(args):
    """执行训练并保存结果"""
    # 创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    exp_name = f"{timestamp}_{args.model}"
    exp_dir = ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"实验名称: {exp_name}")
    print(f"模型: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}\n")
    
    # 加载模型
    model_cfg = MODELS[args.model]
    if model_cfg.endswith(".pt"):
        model = YOLO(model_cfg)
    else:
        # 自定义配置，加载预训练权重
        base_weights = "yolo11n-seg.pt" if "n" in args.model or args.model == "eca" else "yolo11m-seg.pt"
        model = YOLO(str(ROOT / model_cfg)).load(base_weights)
    
    # 训练
    results = model.train(
        #data=str(ROOT / "configs/fruit_data_v1.yaml"),
        data=str(ROOT / "configs/data.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=0,
        project=str(exp_dir),
        name="train",
        workers=2,
        exist_ok=True,
        patience=20,  # 100轮训练用更大的patience
        verbose=True,
    )
    
    # 获取模型效率信息
    try:
        import thop
        import torch
        params = sum(p.numel() for p in model.model.parameters())
        params_m = params / 1e6
        dummy_input = torch.randn(1, 3, 640, 640).to(next(model.model.parameters()).device)
        flops, _ = thop.profile(model.model, inputs=(dummy_input,), verbose=False)
        flops_g = flops / 1e9
    except Exception as e:
        print(f"获取效率指标失败: {e}")
        params_m = 0
        flops_g = 0
    
    # 提取关键指标
    rd = results.results_dict
    metrics = {
        "model": args.model,
        "epochs": args.epochs,
        "timestamp": timestamp,
        # 检测指标 (主要)
        "box_mAP50": round(float(rd.get("metrics/mAP50(B)", 0)), 4),
        "box_mAP50-95": round(float(rd.get("metrics/mAP50-95(B)", 0)), 4),
        "precision": round(float(rd.get("metrics/precision(B)", 0)), 4),
        "recall": round(float(rd.get("metrics/recall(B)", 0)), 4),
        # 分割指标 (次要)
        "mask_mAP50": round(float(rd.get("metrics/mAP50(M)", 0)), 4),
        "mask_mAP50-95": round(float(rd.get("metrics/mAP50-95(M)", 0)), 4),
        # 效率指标
        "params_M": round(params_m, 2),
        "flops_G": round(flops_g, 1),
    }
    
    # === 获取每类准确率 ===
    try:
        # 验证并获取每类指标
        val_results = model.val(data=str(ROOT / "configs/data.yaml"), verbose=False)
        class_names = val_results.names
        
        # 每类 AP50 和 AP50-95
        per_class = {}
        if hasattr(val_results, 'box') and val_results.box.ap50 is not None:
            for i, name in class_names.items():
                per_class[name] = {
                    "AP50": round(float(val_results.box.ap50[i]), 4),
                    "AP50-95": round(float(val_results.box.ap[i]), 4),
                }
        metrics["per_class"] = per_class
        
        # 打印每类结果
        print(f"\n{'='*60}")
        print("每类检测结果 (Box):")
        print(f"{'类别':<15} {'AP50':<10} {'AP50-95':<10}")
        print("-" * 35)
        for name, vals in per_class.items():
            print(f"{name:<15} {vals['AP50']:<10.4f} {vals['AP50-95']:<10.4f}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"获取每类指标失败: {e}")
    
    # 保存指标
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # 打印结果摘要
    print(f"\n{'='*60}")
    print(f"训练完成: {exp_dir}")
    print(f"{'='*60}")
    print(f"检测指标:")
    print(f"  mAP50:     {metrics['box_mAP50']:.4f}")
    print(f"  mAP50-95:  {metrics['box_mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"分割指标:")
    print(f"  Mask mAP50:     {metrics['mask_mAP50']:.4f}")
    print(f"  Mask mAP50-95:  {metrics['mask_mAP50-95']:.4f}")
    print(f"效率:")
    print(f"  Params: {metrics['params_M']}M")
    print(f"  FLOPs:  {metrics['flops_G']}G")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="果实检测模型训练")
    parser.add_argument("--model", type=str, default="baseline", 
                        choices=list(MODELS.keys()),
                        help="模型类型: baseline, eca, teacher, teacher_eca")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小")
    args = parser.parse_args()
    
    train(args)
