import os
import random
import shutil
from torchvision import datasets

# 设置数据集路径
data_dir = '../datasets/lfw'
output_dir = '../datasets/lfw_sampled'  # 随机选择500张图片保存的目录

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 使用 ImageFolder 加载数据集
dataset = datasets.ImageFolder(data_dir)

# 获取数据集中所有图片的路径和标签
all_images = dataset.imgs  # 返回 (image_path, label) 元组的列表

# 随机选择500张图片
sampled_images = random.sample(all_images, 500)

# 将随机选择的500张图片复制到新目录
for img_path, label in sampled_images:
    class_name = os.path.basename(os.path.dirname(img_path))  # 获取类别名称
    output_class_dir = os.path.join(output_dir, class_name)
    
    # 如果类别文件夹不存在，创建文件夹
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
    
    # 复制图片到新的目录
    shutil.copy(img_path, output_class_dir)

print(f"Successfully copied 500 images to {output_dir}")
