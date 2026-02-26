import os
import pandas as pd
import shutil
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import json
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image  # 新增：用PIL处理图片更稳定


def random_augmentation():
    """生成随机的数据增强组合（返回Compose对象）"""
    augmentation_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomRotation(degrees=(-15, 15))
    ]
    num_ops = random.randint(1, 2)
    selected_transforms = random.sample(augmentation_transforms, num_ops)
    return transforms.Compose(selected_transforms)


class CustomDataset(Dataset):
    def __init__(self, data_root, image_names, labels, train=False):
        self.data_root = data_root
        self.image_names = image_names
        self.labels = labels
        self.train = train
        
        # 1. 调整预处理顺序：先Resize，再增强（针对PIL图片）
        self.base_transform = transforms.Compose([
            transforms.Resize((480, 640), antialias=True),  # antialias=True更稳定
            transforms.ToTensor(),  # 新增：PIL→tensor，自动完成(H,W,C)→(C,H,W)和归一化
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.data_root, image_name)
        
        # 2. 安全读取图片（添加异常处理，兼容损坏/不存在的图片）
        try:
            # 优先用PIL读取（避免cv2的编码/通道问题）
            image = Image.open(image_path).convert('RGB')  # 强制转为RGB，避免灰度/透明通道
        except Exception as e:
            print(f"警告：读取图片失败 {image_path}，原因：{str(e)}，使用默认图片替代")
            # 生成默认图片（480x640 RGB）
            image = Image.new('RGB', (640, 480), color=(0, 0, 0))
        
        # 3. 基础预处理（Resize + 转Tensor + 归一化）
        image = self.base_transform(image)  # 输出shape: (3, 480, 640)，值范围0-1
        
        # 4. 训练阶段的随机增强（每次getitem重新生成增强，保证随机性）
        if self.train and torch.rand(1).item() > 0.5:
            aug_transform = random_augmentation()
            image = aug_transform(image)
        
        # 5. 处理标签
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            'image_path': image_path,
            'image': image,
            'label': label
        }


def move_files(image_names, data_path, target_dir):
    """将图片复制到目标目录（添加日志）"""
    for image_name in image_names:
        source = os.path.join(data_path, image_name)
        destination = os.path.join(target_dir, image_name)
        if os.path.isfile(source):
            shutil.copy(source, destination)
        else:
            print(f"文件不存在：{source}")


def split_dataset(mos_file, data_path, test_size=0.3, random_state=42):
    """划分数据集，生成train/test Dataset"""
    # 读取Excel文件（假设第一列是图片名，第二列是MOS评分）
    df = pd.read_excel(mos_file)
    image_names = df.iloc[:, 0].values
    labels = df.iloc[:, 1].values

    # 划分训练/测试集
    train_images, test_images, train_labels, test_labels = train_test_split(
        image_names, labels, test_size=test_size, random_state=random_state)

    # 创建目录
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 复制图片
    move_files(train_images, data_path, train_dir)
    move_files(test_images, data_path, test_dir)

    print("数据集划分完成！\n")

    # 生成Dataset实例
    train_dataset = CustomDataset(train_dir, train_images, train_labels, train=True)
    test_dataset = CustomDataset(test_dir, test_images, test_labels, train=False)

    return train_dataset, test_dataset


def save_dataset(train_dataset, test_dataset, data_path):
    """保存Dataset实例到本地"""
    os.makedirs(os.path.join(data_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'test'), exist_ok=True)

    torch.save(train_dataset, os.path.join(data_path, 'train/train_dataset.pth'))
    torch.save(test_dataset, os.path.join(data_path, 'test/test_dataset.pth'))

    print("Dataset保存完成！\n")


def load_dataset(data_path):
    """加载本地保存的Dataset实例"""
    train_dataset = torch.load(os.path.join(data_path, 'train/train_dataset.pth'))
    test_dataset = torch.load(os.path.join(data_path, 'test/test_dataset.pth'))

    print("Dataset加载完成！\n")
    return train_dataset, test_dataset