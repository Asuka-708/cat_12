# 路径操作
import os
# 文件操作
import shutil
# torch相关
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.autograd import Variable
# 日志
import logging
import random
import numpy as np
import matplotlib.pyplot as plt # 绘图展示工具
# 计算机视觉cv2
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
# 图像处理
from PIL import Image, ImageEnhance
folder_path = r"data\cat_12\cat_12_train"  # 文件夹路径
for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg'):
        file_path = os.path.join(folder_path, file_name)
        with Image.open(file_path) as img:
            width, height = img.size
            print(f'{file_name}: {width} x {height} pixels')

## 统计训练集各类猫的数目，防止样本不平衡问题。
path_train_list = (r"data\cat_12\train_list.txt")

with open(path_train_list, "r") as f:
    labels = f.readlines()
    labels = [int(i.split()[-1]) for i in labels]


counts = pd.Series(labels).value_counts().sort_index().to_list()
names = [str(i) for i in list(range(12))]
data = list(zip(counts, names))
source = [list(i) for i in data]
for item in source:
    print(item)
    print()

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 转换文件夹格式
train_ratio = 0.9  # 训练集占0.9，验证集占0.1

folder_path = 'data/cat_12_train'  # 原文件夹路径
train_list_path = 'data/cat_12/train_list.txt' # 分割集txt路径

train_paths, train_labels = [], []
train_paths_new = []
# 读取分割集，并进行分割
with open(train_list_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        train_paths.append(line.split('	')[0])
        train_paths_new.append(line.split('	')[0].split('/')[1])
        label = line.split('	')[1]
        train_labels.append(int(line.split('	')[1]))

# 先新建一个文件夹，防止数据错乱
newfolder_name = '\\cat_12_train_new'  # 新文件夹名称
newfolder_path = r"data"
os.makedirs(os.path.join(newfolder_path,newfolder_name), exist_ok=True)

# 一种猫类建立一个文件夹
train_labels_new = list(set(train_labels))
train_labels_new.sort(key=train_labels.index)
for i in train_labels_new:
    newfolder_name = str(i)
    newfolder_path = 'data/cat_12_train_new/'  # 新文件夹路径
    os.makedirs(os.path.join(newfolder_path, newfolder_name), exist_ok=True)

# 先新建一个文件夹，防止数据错乱
newfolder_name = '\\cat_12_val_new'  # 新文件夹名称
newfolder_path = r'data'
os.makedirs(os.path.join(newfolder_path,newfolder_name), exist_ok=True)

# 一种猫类建立一个文件夹
train_labels_new = list(set(train_labels))
train_labels_new.sort(key=train_labels.index)
for i in train_labels_new:
    newfolder_name = str(i)
    newfolder_path = r"data\cat_12_val_new"  # 新文件夹路径
    os.makedirs(os.path.join(newfolder_path, newfolder_name), exist_ok=True)

# 将对应猫片复制入对应猫文件夹
for i in range(0, len(train_paths)):
    if random.uniform(0, 1) < train_ratio:
        src_img_path = 'data/cat_12/' + str(train_paths[i])  # 源图片路径
        cat_type = str(train_labels[i])
        dst_folder_path = "data/cat_12_train_new/" + cat_type  # 目标文件路径
        img_name = os.path.basename(src_img_path)  # 获取图片文件名
        dst_path = os.path.join(dst_folder_path, img_name)  # 构造目标文件路径
        shutil.copy(src_img_path, dst_path)  # 复制
    else:
        src_img_path = 'data/cat_12/' + str(train_paths[i])  # 源图片路径
        cat_type = str(train_labels[i])
        dst_folder_path = "data/cat_12_val_new/" + cat_type  # 目标文件路径
        img_name = os.path.basename(src_img_path)  # 获取图片文件名
        dst_path = os.path.join(dst_folder_path, img_name)  # 构造目标文件路径
        shutil.copy(src_img_path, dst_path)  # 复制


path = 'data\cat_12_val_new'
file_list = os.listdir(path)

for file in file_list:
    # 补0 4表示补0后名字共4位 针对imagnet-1000足以
    filename = file.zfill(4)
    # print(filename)
    new_name = ''.join(filename)
    os.rename(path + '\\' + file, path + '\\' + new_name)
path = 'data\cat_12_train_new'
file_list = os.listdir(path)

for file in file_list:
    # 补0 4表示补0后名字共4位 针对imagnet-1000足以
    filename = file.zfill(4)
    # print(filename)
    new_name = ''.join(filename)
    os.rename(path + '\\' + file, path + '\\' + new_name)

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
    # transforms.ColorJitter 改变图像的属性：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)。
    # 数值表示上下浮动的百分比。比如：0.05，原亮度100 ->(95,105)范围内。
    transforms.Resize((256, 256)),  # 缩放到指定大小
    transforms.CenterCrop(224),   # 中心随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),  # 0.5 概率水平翻转
    transforms.RandomVerticalFlip(p=0.5),  # 0.5 概率垂直翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=0, std=1),  # 归一化
])

# 加载数据集
folder_path = 'data\cat_12_train_new'# 文件夹路径
dataset = ImageFolder(folder_path, transform=transform)
print(dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# 随机展示3张图
fig = plt.figure(figsize=(16, 16))
for i, (img, label) in enumerate(dataloader):
    if i >= 3:
        break
    img = img.squeeze().numpy().transpose((1, 2, 0))  # 将张量转换为图像
    ax = fig.add_subplot(1, 3, i+1)
    ax.imshow(img)
    ax.set_title(f'Label: {dataset.classes[label.item()]}')
plt.show()