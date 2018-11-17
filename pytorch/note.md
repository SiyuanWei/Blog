# PYTORCH CNN

pytorch 主要用于深度学习 张量计算的库 FaceBook C++计算后端 python接口

### 封装
对于一个神经网络 封装的层次大概包括基本的网络层（Layer）,由多个网络层构成的模块（Block），以及最终的模型
```
import torch
class Layer(toch.nn.Module):
      #卷积 反卷积 空洞卷积 分组卷积 
      #正则化
      #激活函数
      #其他矩阵运算的操作
      pass
class Block(torch.nn.Module):
      pass
class Model(torch.nn.Module):
      pass
```

对于某个由多个网络层 激活函数 正则化层组成的Block 在PYtorch中应该这样构造：
```
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
    #定义网络层ConvLayer 正则化 batchnorm
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

```
在自定义的Module中，应当继承torch.nn.Module,构造方法通常的传入参数包括输入通道数目（inchannel），输出通道数目(outchannel),
以及一些方便你调整模块的超参数（hyper parameter）.
* 1在构造方法(init）当中，你应当定义你的Block中应当包含哪些layer层，并根据需要定义这些层的超参数
* 2在前向传播的方法（forward）中,你应当定义输入变量（x）经过何种顺序（**一般而言，1*1卷积用于调整channel数目，
n*n卷积用来获取感受野中的信息，并且后面介入正则化层防止过拟合（batchnorm），再接入激活函数（relu,sigmoid.LRelu,SRelu）
以拟合非线性函数），经过哪些操作,最终获得输出
* 3前向传播函数（forward）再底层实现上类似于python类中的调用函数（__call__），即先通过构造函数定义网络层，再调用foward
函数以进行网络的前向传播

### 动态图特性
相比tensorflow的静态图特性来说，pytorch是一个动态图框架，前向传播函数在每一次调用时，都生成了一次前向网络传播的计算逻辑
，即整个网络流图是动态刷星的。理论而言，静态图框架比动态图框架更快，因为预先将网络层以及整个模型的计算逻辑都定义为一张数
据流图，在运行时可以避免重复生成中间变量和计算逻辑。但是在实际使用中，两者工作成本却差不多，这主要是由于：
* 1.pytorch的简洁封装以及和其他库的易连接性，使得pytorch在实现模型的成本上要更简单易懂，容易上手，并且具有可观的速度
* 2.tensorflow的静态图特性，使得在编写网络时调整超参数，网络模块化，更为麻烦，因为其严谨的静态特性，所以以至于模型中的
循环操作都需要借助于tensorflow的循环函数来实现
* 3.tensorflow的版本迭代有很多地方不向下兼容，封装有些杂乱，列如在数据输入接口方面，尽管十分齐全，但是各种数据输入的方法
并不十分统一，彼此适用的场景也不尽相同

### 数据输入
pytorch中的数据输入需要你按照自己的需求定义一个类
* 1 继承Dataset类
* 2 在构造方法中，一般而言，是通过你的输入变量来获取你的数据集源文件地址列表（Images and lables）
* 3 该类覆盖（\__len__）方法和（\__getitem___）方法，使得这个类成为一个容器（Container），以获取模型的长度，并且可以按
Key取值，迭代获取数据。简单而言，可以理解为一个数据集容器，根据数据源文件地址迭代获取数据，经过预处理，获得张量
* 4 一般而言，你需要输入一个transform变量，这个变量应当是一个负责数据预处理的实例，在\__getitem__函数运行的时候，通过
transform进行数据预处理和数据增强
```
from torch.utils.data import Dataset
class BaseDataset(Dataset):

    def __init__(self, img_files, label_files, transform=None, num_classes=2, is_transform=False, ignore_lable=255, label_map=None):
        self.img_files = img_files
        self.label_files = label_files
        self.num_classes = num_classes
        self.is_transform = is_transform
        self.ignore_label = ignore_lable
        self.transform=transform
        if label_map:
            self.idmap_func = np.vectorize(label_map.get)
        else:
            self.idmap_func = None


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image,label = self.img_files[idx],self.label_files[idx]
        image,label = io.imread(image),io.imread(label)

        if self.idmap_func:
            label = self.idmap_func(label)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

def loader(dataset, batch_size,  shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)
    return loader
```

### 数据预处理和数据增强
对于各类模型来说，数据预处理和数据增强都十分重要，对于提高模型的泛化性能十分十分重要
例如，使用pytorch实现一个图像分割的神经网络模型，你需要在实现的transform操作包括：
* 1. 图像增强：resize 以匹配网络输入的张量形状需求，rescale 放缩图像  以提高网络对多尺度的适应能力和泛化能力
RandomCrop 随机裁剪，RandomFlip,随机翻转，RandomGaussianBlur 随机高斯模糊，使用随机参数高斯模糊图像，减少图像中的噪声
Whitening，图像白化，对一张图像中的每个像素点进行标准化（减去图像总像素的均值再除以标准差），CLAHE（限制直方图均衡化），
提升图像的锐度，使得边缘特征更加明显，其他图像增强操作还包括随机扭曲、随机旋转，这两类在医学图像处理问题上使用较多，原因
在于扭曲和旋转操作不影响 医学中各类细胞、病理部位的特征丢失或错误
* 2. 数据预处理：OneHot,toTensor
```
from __future__ import print_function, division

import numpy as np
from skimage import transform
from torch.distributions.one_hot_categorical import *

class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        new_h, new_w = self.output_size

        #skimage transform resize,The order of interpolation.
        # The order has to be in the range 0-5:
        # * 0: Nearest-neighbor * 1: Bi-linear (default) * 2: Bi-quadratic * 3: Bi-cubic * 4: Bi-quartic * 5: Bi-quintic
        image = transform.resize(image, (new_h, new_w),order=1)
        label = transform.resize(label, (new_h, new_w),order=0)


        return {'image': image, 'label': label}
```
### 网络设计调参
实际使用中，如果不做算法创新，往往是借鉴别人的网络结构，迁移到自己的应用场景中，进行调参
对于学习率这类超参数，往往采用经验参数就可以
对于网络结构中的输入输出通道以及模块数目，连接方式，建议阅读论文，参考经典网络结构（ResNet,ResNext,DenseNet,Inception,
UNet,FCN,DeeplabV3+）
注意，不同框架的padding方法不一样，例如同样输入padding=[1,1],tensorflow是在图片左上各补齐一行，pytorch是在网络上下左右全
部补齐一行,对于pytorch,
>>> output_shape = (image_shape-filter_shape+2*padding)/stride + 1

## 训练与可视化
训练时需要考虑的是
* 1.梯度下降函数 简单的问题一般采用SGD或者AdamGd,现在大部分图像神经网络采用待用动量的随机梯度下降MomentumGD
,学习率随机迭代次数的增加逐渐衰减，防止loss震荡过大无法收敛
* 2.损失函数  CrossEntropy 交叉熵 MSE 均方误差  各类损失函数 详情阅读最新的论文
* 3.输入的批量 batchsize 一般来说 机器性能不封顶的情况下  batchsize越大，训练越快，batchnorm效果也较好，然而，在现在很多
端对端训练的问题中，batch越大，内存开销也越大，如果使用的是gpu，显存开销也越大
* 4.评价指标 很多情况 对与模型的评价指标并非是简单的准确率 还可能包括IOU 平均交并比 Dice's coefficient 
* 5.损失 网络参数可视化 tensoboard tensorboardX
```
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import datetime
from tensorboardX import SummaryWriter
from utils.eval import *
from torch.utils.checkpoint import checkpoint_sequential

def train(model, data_loader, optimizer, criterion,
          batch_size=10, epochs=10,eval_func = IOU, eval_step =10, log_params = False, device='cpu', save_model = True):

    writer = SummaryWriter()
    model.to(device)
    model.train()

    assert device in ['cpu','cuda'],'device should be cpu or cuda'
    if device=='cuda':
        log_device = 'cpu'
    else:
        log_device = 'cuda'

    niter = 0
    eval_results = []
    for epoch in range(epochs):
        for batch_idx, sample in enumerate(data_loader):
            optimizer.zero_grad()
            input, label = sample['image'].to(device),sample['label'].to(device)
            input.requires_grad_(requires_grad=False)
            output = model(input)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            niter += batch_size

            writer.add_histogram('loss', loss.clone().to('cpu').data.numpy(), niter)

            if (batch_idx+1)*batch_size % eval_step == 0:
                #
                eval = eval_func(output, label)
                info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Eval: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(input), len(data_loader.dataset),
                           100. * (batch_idx+1) / len(data_loader), loss.item(),eval)
                print(info)
                writer.add_text('Text',info)
                del info
                writer.add_histogram('eval', eval, niter)
                eval_results.append(float(eval))

```

### 尽管如此
tensorflow工业界别的运行速度和部署，和丰富的底层接口使得tensorflow是工业界应用的首选，如果要学习深度学习和一门张量计算框
架的话，建议从pytorch入门，再接触tensoflow




