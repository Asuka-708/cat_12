import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageDraw
from tqdm import tqdm


# 对应超参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
LR = 0.005
EPOCH = 100
BTACH_SIZE = 32
train_root = "data/cat_12_train_new"


def add_gaussian_noise(x):
    # x 是一个 float 类型的 tensor，可以直接进行浮点数运算
    noise = torch.randn_like(x) * 0.02  # 添加标准差为 0.02 的高斯噪声
    return x + noise


def add_random_mask(image, mask_size=(250, 250), p=0.5):
    """
    在图像上随机添加一个矩形遮挡。

    参数:
        image (PIL Image): 输入图像。
        mask_size (tuple): 遮挡矩形的大小，格式为(width, height)。
        p (float): 应用遮挡的概率。

    返回:
        PIL Image: 应用了遮挡的图像或原始图像（如果未应用遮挡）。
    """
    if random.random() > p:
        return image

    draw = ImageDraw.Draw(image)
    width, height = image.size
    left = random.randint(0, width - mask_size[0])
    top = random.randint(0, height - mask_size[1])
    right = left + mask_size[0]
    bottom = top + mask_size[1]
    draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))  # 假设遮挡颜色为黑色
    return image


# 定义你的train_transform，包含随机遮挡
train_transform = transforms.Compose([
    transforms.Resize(256),                                                        ##将图片转化为256 x 256大小
    transforms.CenterCrop(256),                                                     #踩区中间256*256的大小
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.0)),      #实现随机切割
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),       #改变亮度，对比度，饱和度，色调
    transforms.RandomHorizontalFlip(p=0.5),                                          #随即水平翻转
    transforms.RandomVerticalFlip(p=0.5),                                            #随机竖直翻转
    transforms.ToTensor(),
    transforms.Lambda(lambda x: add_gaussian_noise(x)),                              #添加高斯噪音
    transforms.Lambda(lambda x: add_random_mask(Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)),
                                                mask_size=(50, 50), p=0.5)),         #实现随机去除
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757]),          #实现归一化
])


val_transform = transforms.Compose([                                                #对测试集的简单处理
    transforms.Resize((256, 256)),  # 缩放到指定大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757]),  # 归一化
])

train_d = 'data\cat_12_train_new'
train_data = ImageFolder(train_d,transform=train_transform)


# 使用测试集的样本索引生成验证集valid_data。
valid_data = 'data\cat_12_val_new'
val_dataset = ImageFolder(valid_data,transform=val_transform)
# 训练数据集加载
train_set = torch.utils.data.DataLoader(
    train_data,
    batch_size=BTACH_SIZE,
    shuffle=True
)

# 测试集加载
test_set = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BTACH_SIZE,
    shuffle=True
)


# 训练
def train(model1, device, dataset, optimizer1, epoch1):         #model是神经网络，device为GPU，dataset是训练数据集，optimizer1为优化器，epoch1为训练轮数
    global loss
    model1.train()

    correct = 0
    all_len = 0
    # 'tqdm'是一个用于显示进度条的库，它接受任何可迭代对象，并在遍历这个可迭代对象时显示一个进度条。
    for i, (x, y) in tqdm(enumerate(dataset)):      #i为索引 x为张量图片 y为x对应的标签   这个里我们用0到11的数来代替猫的种类，这在UI界面时会细说
        x, y = x.to(device), y.to(device)       #转移到GPU
        optimizer1.zero_grad()                  #梯度清零防止梯度爆炸等其他问题
        output = model1(x)                      #返回[batch_size, num_classes]
        pred = output.max(1, keepdim=True)[1]       #1为要查询第二个维度，获取与图片最匹配的类别的索引
        correct += pred.eq(y.view_as(pred)).sum().item()   #逐元素比较操作，返回一个布尔张量，其中对应位置的元素为True如果预测和真实标签相等，否则为False
        all_len += len(x)                           #获得数据量以便求出训练真实值
        loss = nn.CrossEntropyLoss()(output, y)             #交叉熵回归函数计算残以及对图片进行分类
        loss.backward()                             #反向传播计算梯度
        optimizer1.step()

    print(f"第 {epoch1} 次训练的Train真实：{100. * correct / all_len:.2f}%")

# 测试机验证
def vaild(model, device, dataset):
    model.eval()                            #切换为评估模式
    global loss
    correct = 0
    test_loss = 0
    all_len = 0
    with torch.no_grad():           #确保在验证过程中不会计算梯度，从而节省内存和计算资源。
        for i, (x, target) in enumerate(dataset):
            x, target = x.to(device), target.to(device)

            output = model(x)
            loss = nn.CrossEntropyLoss()(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_len += len(x)
    print(f"Test 真实：{100. * correct / all_len:.2f}%")
    return 100. * correct / all_len




model_1 = torchvision.models.resnet50(pretrained=True)  # weights='ResNet50_Weights.DEFAULT'
model_1.fc = nn.Sequential(
    nn.Linear(2048, 12)                 #修改输出层确保分类为12种
)

model_1.to(DEVICE)
optimizer = optim.SGD(model_1.parameters(), lr=LR, momentum=0.09,weight_decay = 3e-4 )      #优化器的设置

max_accuracy = 90.0  # 设定保存模型的阈值
best_model = None

for epoch in range(1, EPOCH + 1):
    train(model_1, DEVICE, train_set, optimizer, epoch)
    accu = vaild(model_1, DEVICE, test_set)
    if accu > max_accuracy:
        max_accuracy = accu
        best_model = model_1.state_dict()  # 或者使用 torch.save() 保存整个模型

# 保存最优模型
torch.save(best_model, r"best_model_train1.pth")

