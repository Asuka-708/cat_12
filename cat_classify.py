from collections import Counter#一种字典的子类，将输入转换成字典，输入的每一个元素作为字典的键，每个元素的个数作为值，主要作用是方便计算
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.utils.data import WeightedRandomSampler, sampler  # w即为权重，训练中会根据权重选择权重较大的样本进行训练
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm#可用于创建一个模拟进度条，以便清楚看到训练的进度

#超参
Device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')#设置设备是否可用gpu的参数
LR=0.01#学习率
EPOCH=60#训练轮数
Batch_size=64#批量大小
train_root='data/cat_12_train_new'#训练网络的数据集所在位置

batch_size=8#批量大小

#套用方法，只有先把数据导入并处理成能够使神经网络识别的类型才能进行下面的部署。数据加载及处理
train_transform=transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(224,scale=(0.6,1.0),ratio=(0.8,1.0)),#从图像中裁剪出一个区域。缩放至224*224像素，scale参数表示裁剪的区域是原图片多少到多少的比例，ratio参数代表裁剪的区域的宽高比
                                    transforms.RandomHorizontalFlip(),#有50%概率对图片进行水平翻转，以便于机器学习图像处理的对称性，提高泛化能力
                                    transforms.RandomVerticalFlip(),
                                    transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0),#随机改变图像的亮度度，范围在1-0.5~1+0.5，亮度，饱和度，色调不变（括号内一一对应）
                                    transforms.ColorJitter(brightness=0,contrast=0.5,saturation=0,hue=0),#与上一行一样不过只改变了对比度

                                    transforms.ToTensor(),#将PIL图像或数组转化为张量，并且缩放到0~1之间
                                    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),#对图片进行标准化，意外着图像数据会被减去0.5再除以0.5，最后得到的数据范围在-1~1之间，属于预处理步骤，利于模型更快地收敛
                                    ])
eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 缩放到指定大小
    transforms.CenterCrop(196),   # 中心随机裁剪
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean =[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757]),  # 归一化
])

#图像读取转换
all_data=torchvision.datasets.ImageFolder(
    root=train_root,
    transform=train_transform
)
# test__data=torchvision.datasets.ImageFolder(
#     root='cat_train_path2/cat_12_val_new',
#     transform=eval_transform
# )

print(len(all_data))
print(all_data.class_to_idx)#返回子文件夹的索引

#计算每个样本类别的数量
class_counts=Counter(all_data.targets)#直接以all_data里的路径进行计数
print(class_counts)
#计算每个类别的样本权重
weights=[1.0/class_counts[class_idx] for class_idx in all_data.targets]
# print(weights)#该权重是按照该类别出现的频率的倒数计算出来的

#创建一个WeighedRandomSampler对象，这个对象是pytorch的数据加载工具中的一个功能，可以应用于从带有权重的列表进行采样
#replacement=True表示可以进行有放回的采样
sampler = WeightedRandomSampler(weights, len(all_data), replacement=True)#权重为标准进行采样，数量为len(...)，最后一个参数为是否放回.得到的sampler只是原数据索引的一部分，是由权重随机选取的

#使用采样器进行数据集划分，从整体数据集all_data中根据sampler的采样结果选取部分数据，得到的作为训练数据集train——data
train_data=torch.utils.data.Subset(all_data,list(sampler))#Subset的作用为在原始数据的基础下创建一个子集，其基于指定的索引列表（即sampler）选取出的。Subset 常常与 torch.utils.data.Dataset 一起使用，用于创建可迭代的数据加载器（DataLoader），以便在训练神经网络时按批次加载数据。
#将sampler转化为一个列表，其中的元素是采样得到的样本索引
sampler_indices=list(sampler)
# print(sampler_indices)

#创建一个未被采样的索引列表作为后续的测试集参数
valid_indices=[idx for idx in range(len(all_data)) if idx not in sampler_indices]
# print(valid_indices)

#以valid_indices创建测试集
valid_data=torch.utils.data.Subset(train_data,valid_indices)

#训练数据集加载
train_set=torch.utils.data.DataLoader(
    train_data,
    batch_size=Batch_size,
    shuffle=True
)
#测试数据集加载
test_set=torch.utils.data.DataLoader(
    valid_data,
    batch_size=Batch_size,
    shuffle=True
)

#模型训练与预测

#训练
def train(model1,device,dataset,optimizer1,epoch1):
    global loss#生命loss是一个全局变量，用于计算整个过程中的损失值
    model1.train()

    correct=0
    all_len=0#分别代表计数正确处理样本以及处理的样本个数

    for i,(x,y) in tqdm(enumerate(dataset)):#enumerate会同时返回在数据集中的索引i以及对应数据x，y（输入与对应的目标或标签）
       x,y=x.to(device),y.to(device)#将x，y分别对应到相应设备配置中（GPU或CPU）
       optimizer1.zero_grad()#优化器中的梯度清零，不加的话默认为不清零，但实际上的梯度是需要每次迭代之后清零才是真是有效的
       output=model1(x)#将输入数据x传递到模型中，并返回输出output
       predict=output.max(1,keepdim=True)[1]#此行用来计算模型的预测，max(1, keepdim=True)返回的是每个输出中的索引最大值，这个最大值就是预测结果，【1】就是提取索引
       correct+=predict.eq(y.view_as(predict)).sum().item()#简单将作用就是把预测对的数量统计了下来
       #具体操作为通过eq函数（对比两个张量中的每个值是否相等，是则为True即1，反之为0），然后求出对比后1的总和（对比后得到的形式也为张量），即为预测正确的数量
       all_len+=len(x)#求处理的总样本数量，len（x）是返回的每个批次中的样本数量
       loss=nn.CrossEntropyLoss()(output,y)#计算交叉熵损失，即通过模型预测的结果outpu与原标签y之间的损失
       loss.backward()#反向传播，计算损失相对于模型参数的梯度
       optimizer1.step()#使用优化器更新模型的权重。通过之前计算的梯度调整模型的参数

    print(f'第{epoch1}次训练的Train真实：{100.*correct/all_len:.2f}%')#输出预测的准确率

#测试机验证
def valid(model,device,dataest):
    model.eval()#将模型设置为评估模式
    global loss#声明loss是全局变量
    correct=0
    all_len=0#分别代表两个计数样本处理正确与原样本的数量变量
    test_loss=0#测试的损失值
    with torch.no_grad():#确保验证集不会计算梯度（没必要且省时省空间）
        for i,(x,target) in enumerate(dataest):#作用为：i返回数据索引，（x,targer）分别为输入数据与其真实标签。
            x,target=x.to(device),target.to(device)#将对应的数据传入相应处理设备中
            output=model(x)#将x通过模型的处理得到output
            loss=nn.CrossEntropyLoss()(output,target)#计算测试与真是的交叉熵
            test_loss+=loss.item()#将当前批次的损失累加到test_loss中
            pred=output.argmax(dim=1,keepdim=True)#argmax不论输入的是几维张量都会先转换为一维（相当于一个列表），dim（不论dim是几）就是表示维度会消失。dim=0表示行不变只从列中比较找出值较大的索引，等于1时反之
            correct+=pred.eq(target.view_as(pred)).sum().item()#计数预测正确的数量
            all_len+=len(x)#处理过的样本的计数
    print(f'Test真实：{100.*correct/all_len:.2f}%')
    return 100.*correct/all_len

#ResNet50迁移学习
model1=torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V1)#导入现有模型（要不自己写，要不导入不然无法进行训练）。pretrained=True表示加载与训练的权重，无需自己再设置参数
model1.fc=nn.Sequential(
    nn.Linear(2048,12)#表示分类任务为12个（输出特征）。输入特征为2048个一般自己写的网络需要提前算出次数（将最后的结果拉成一维得到的数字就是），这里是因为导入的网络原本就是此
)#激活非线性
model1.to(Device)
optimizer=optim.SGD(model1.parameters(),lr=LR,momentum=0.09)#导入SGD优化器(随机梯度下降)，并设置参数

#模型的训练和保存
max_accuracy=80.0#保存模型的阈值
best_model=None

for epoch in range(1,EPOCH+1):
    train(model1,Device,train_set,optimizer,epoch)
    accu=valid(model1,Device,test_set)
    if accu>max_accuracy:
        max_accuracy=accu
        best_model=model1.state_dict()
torch.save(best_model,r'cat_net.pth5')#以类似字典的形式保存训练的权重


import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision
import pandas as pd
import os
from PIL import Image

num_classes = 12

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 缩放到指定大小
    transforms.CenterCrop(256),
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757]),  # 归一化
])


def GetFiles(file_dir, file_type, IsCurrent=False):
    file_list = []
    for parent, dirnames, filenames in os.walk(file_dir):
        for filename in filenames:
            if filename.endswith(('.%s' % file_type)):  # 判断文件类型
                file_list.append(filename)
        if IsCurrent == True:
            break
    return file_list


def model():
    model_1 = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model_1.fc = nn.Sequential(nn.Linear(2048, 12))
    model_1.to(device)
    model_1.load_state_dict(torch.load(r'cat_net.pth5'))
    model_1.eval()
    return model_1


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model().to(device)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    labels = []
    names = GetFiles('data/cat_12/cat_12_test', 'jpg')
    error_names = []
    for i in names:
        img_path = 'data/cat_12/cat_12_test/{}'.format(i)
        try:
            img = Image.open(img_path)
            img_ = test_transform(img).unsqueeze(0)
            img_ = img_.to(device)
            outputs = model(img_)
            indices = torch.max(outputs, 1)
            percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            perc = percentage[int(indices)].item()
            result = class_names[indices]
            labels.append(result)
        except:
            labels.append('5')
            error_names.append(img_path)
    df1 = pd.DataFrame(names)
    df2 = pd.DataFrame(labels)
    df = pd.concat([df1, df2],axis=1,join='outer')
    df.to_csv('result4.csv', header=False, index=False)
    print("Done!")