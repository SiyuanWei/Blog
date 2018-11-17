import torch
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import torch.nn as nn
#对于一类计算iou

def _iou(pred, label,cls):
    #pred label 展开的tensor

    tp_idxs = (pred == cls)
    cls_idxs = (label == cls)

    intersection = (tp_idxs[cls_idxs]).long().sum().data
    union = tp_idxs.long().sum().data + cls_idxs.long().sum().data - intersection
    return float(intersection) / float(union)

def IOU(pred, label,cls=1):
    # pred 概率tensor label是Tensor
    # pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    label = label.view(-1)

    return _iou(pred,label,cls)

def mIOU(pred,label,cls_list):
    #cls_list应该是包含计算miou的对象Id的列表
    sum=0.
    IOUs=[]
    pred = pred.view(-1)
    label = label.view(-1)
    for cls in cls_list:
        iou = _iou(pred,label,cls)
        IOUs.append(iou)
        sum +=iou
    miou = sum/float(len(cls_list))
    return miou,IOUs


def eval(model,data_loader,device,eval_func=IOU):
    with torch.no_grad():
        model.eval()
        eval_results=[]
        for sample in data_loader:
            image, label = sample['image'].to(device), sample['label'].to(device)
            pred = model(image)
            temp=eval_func(pred,label)
            eval_results.append(temp)
            print('single image eval:{:.6f} '.format(temp))
            del temp

        print('Average image eval:{:.6f} '.format(sum(eval_results)/len(eval_results)))



# 模型显存占用监测函数
# model：输入的模型
# input：实际中需要输入的Tensor变量
# type_size 默认为 4 默认类型为 float32

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'


          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))
if __name__ == '__main__':
    label1 = './1.png'
    label2 = './2.png'

    label1, label2 = torch.from_numpy(io.imread(label1, mode='L')).long(),\
                     torch.from_numpy(io.imread(label2, mode='L')).long()

    iou = IOU(label1,label2)
    print()
