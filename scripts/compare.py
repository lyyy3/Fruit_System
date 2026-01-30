"""
对比所有实验结果
用法: python scripts/compare.py
"""
import json
from pathlib import Path

def compare():
    """读取所有实验的metrics.json并生成对比表格"""
    exp_dir = Path(__file__).parent.parent / "experiments"
    
    if not exp_dir.exists():
        print("未找到experiments目录")
        return
    
    results = []
    
    for exp in sorted(exp_dir.iterdir()):
        if not exp.is_dir():
            continue
        metrics_file = exp / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, encoding="utf-8") as f:
                m = json.load(f)
                results.append(m)
    
    if not results:
        print("未找到任何实验结果")
        return
    
    # 打印表头
    print("\n" + "="*120)
    print(f"{'实验时间':<15} {'模型':<12} {'Epochs':<8} "
          f"{'mAP50':<8} {'mAP50-95':<10} {'P':<8} {'R':<8} "
          f"{'Mask50':<8} {'Mask50-95':<10} {'Params':<8} {'FLOPs':<8}")
    print("="*120)
    
    # 打印每行
    for m in results:
        print(f"{m.get('timestamp', '-'):<15} "
              f"{m.get('model', '-'):<12} "
              f"{m.get('epochs', '-'):<8} "
              f"{m.get('box_mAP50', 0):<8.4f} "
              f"{m.get('box_mAP50-95', 0):<10.4f} "
              f"{m.get('precision', 0):<8.4f} "
              f"{m.get('recall', 0):<8.4f} "
              f"{m.get('mask_mAP50', 0):<8.4f} "
              f"{m.get('mask_mAP50-95', 0):<10.4f} "
              f"{m.get('params_M', 0):<8} "
              f"{m.get('flops_G', 0):<8}")
    
    print("="*120)
    
    # 导出CSV
    csv_path = exp_dir / "comparison.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = ["timestamp", "model", "epochs", "box_mAP50", "box_mAP50-95", 
                   "precision", "recall", "mask_mAP50", "mask_mAP50-95", 
                   "params_M", "flops_G"]
        f.write(",".join(headers) + "\n")
        for m in results:
            row = [str(m.get(h, "")) for h in headers]
            f.write(",".join(row) + "\n")
    
    print(f"\n对比结果已导出: {csv_path}")


if __name__ == "__main__":
    compare()
