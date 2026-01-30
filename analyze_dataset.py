import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from PIL import Image

# === 配置 ===
DATASET_PATH = r"datasets\fruit_dataset_v2" # 你的数据集路径
LABEL_DIR = os.path.join(DATASET_PATH, "train", "labels")
IMAGE_DIR = os.path.join(DATASET_PATH, "train", "images")
CLASSES = ['apple', 'banana', 'citrus', 'grape', 'kiwi', 'strawberry'] # 你的类别

def analyze_yolo_labels():
    print(f"正在分析数据集: {LABEL_DIR} ...")
    
    stats = []
    txt_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    
    if not txt_files:
        print("❌ 错误：找不到任何 .txt 标签文件！请检查路径。")
        return

    for txt_file in tqdm(txt_files):
        # 尝试找到对应的图片以获取尺寸（假设是 jpg）
        img_name = os.path.basename(txt_file).replace('.txt', '.jpg')
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # 默认尺寸 (如果找不到图片)
        img_w, img_h = 640, 640
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                cls_id = int(parts[0])
                
                # YOLO Seg 格式: class x1 y1 x2 y2 ... (归一化多边形)
                # 我们需要算出外接矩形 (Bounding Box) 来做统计
                coords = [float(x) for x in parts[1:]]
                xs = coords[0::2]
                ys = coords[1::2]
                
                # 算出归一化的宽高
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                
                # 中心点
                cx = (max(xs) + min(xs)) / 2
                cy = (max(ys) + min(ys)) / 2
                
                stats.append({
                    "class_id": cls_id,
                    "class_name": CLASSES[cls_id],
                    "width": w * img_w,   # 绝对像素宽度
                    "height": h * img_h,  # 绝对像素高度
                    "area": (w * img_w) * (h * img_h), # 绝对像素面积
                    "cx": cx,
                    "cy": cy,
                    "aspect_ratio": w / h if h > 0 else 0
                })

    df = pd.DataFrame(stats)
    print(f"分析完成！共统计到 {len(df)} 个实例。")
    return df

def plot_charts(df):
    sns.set_theme(style="whitegrid")
    save_dir = "dataset_analysis"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 类别数量分布 (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='class_name', order=CLASSES)
    plt.title("Class Distribution (Instances)")
    plt.savefig(os.path.join(save_dir, "1_class_dist.png"))
    plt.close()

    # 2. 目标尺寸分布 (Box Plot)
    # 用根号面积代表“尺寸大小”，直观一点
    df['size_sqrt'] = df['area'] ** 0.5
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='class_name', y='size_sqrt', order=CLASSES)
    plt.title("Object Size Distribution (sqrt(Area))")
    plt.ylabel("Size (pixels)")
    plt.savefig(os.path.join(save_dir, "2_object_size.png"))
    plt.close()

    # 3. 宽高比分布 (Scatter)
    plt.figure(figsize=(8, 8))
    # 只取前2000个点画图，不然太卡
    sample = df.sample(min(2000, len(df)))
    sns.scatterplot(data=sample, x='width', y='height', hue='class_name')
    plt.plot([0, 640], [0, 640], 'k--', alpha=0.5) # 对角线
    plt.title("Width vs Height (Aspect Ratio)")
    plt.savefig(os.path.join(save_dir, "3_aspect_ratio.png"))
    plt.close()

    # 4. 位置热力图 (Heatmap)
    plt.figure(figsize=(8, 8))
    plt.hist2d(df['cx'], df['cy'], bins=50, cmap='inferno')
    plt.xlim(0, 1)
    plt.ylim(1, 0) # 图像坐标系 Y 轴向下
    plt.title("Object Location Heatmap")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "4_location_heatmap.png"))
    plt.close()
    
    print(f">>> 图表已保存至 {save_dir} 文件夹")

if __name__ == "__main__":
    df = analyze_yolo_labels()
    if df is not None:
        plot_charts(df)