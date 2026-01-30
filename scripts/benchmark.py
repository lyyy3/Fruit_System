"""
性能基准测试
用法: python scripts/benchmark.py --model baseline
"""
import sys
import time
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def benchmark(model_path: str, imgsz: int = 640, warmup: int = 50, runs: int = 200):
    """测试模型FPS"""
    print(f"\n测试模型: {model_path}")
    print(f"图像大小: {imgsz}x{imgsz}")
    print(f"预热轮数: {warmup}, 测试轮数: {runs}")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取参数量和FLOPs
    info = model.info(verbose=False)
    params_m = info[0] / 1e6 if info else 0
    flops_g = info[1] if info and len(info) > 1 else 0
    
    print(f"参数量: {params_m:.2f}M")
    print(f"FLOPs: {flops_g:.1f}G")
    
    # FPS测试
    if torch.cuda.is_available():
        device = "cuda:0"
        dummy = torch.randn(1, 3, imgsz, imgsz).cuda()
        model.model.eval().cuda()
        
        # 预热
        print(f"\n预热中...")
        for _ in range(warmup):
            model.model(dummy)
        torch.cuda.synchronize()
        
        # 测速
        print(f"测速中...")
        t0 = time.time()
        for _ in range(runs):
            model.model(dummy)
        torch.cuda.synchronize()
        
        fps = runs / (time.time() - t0)
        print(f"\nFPS: {fps:.1f}")
    else:
        print("未检测到GPU，跳过FPS测试")
        fps = 0
    
    return {
        "params_M": round(params_m, 2),
        "flops_G": round(flops_g, 1),
        "fps": round(fps, 1)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt",
                        help="模型路径")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    
    benchmark(args.model, args.imgsz, args.warmup, args.runs)
