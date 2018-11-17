import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from input.DataProvider import *
from input.DatasetConfig import *
from input.preprocess import *
from utils.eval import *
from utils.train import *


class CrossEntropyLoss2D(nn.Module):
    def __init__(self, size_average=True):
        super(CrossEntropyLoss2D, self).__init__()
        self.nll_loss = nn.NLLLoss(size_average=size_average)

    def forward(self, outputs, targets):
        return self.nll_loss(F.log_softmax(outputs), targets)

class UNetConvBlock(nn.Module):
    def __init__(self, input_nch, output_nch, kernel_size=3, activation=lambda x:F.relu(x,inplace=True), use_batchnorm=True, same_conv=True):
        super(UNetConvBlock, self).__init__()
        padding = kernel_size // 2 if same_conv else 0  # only support odd kernel
        self.conv0 = nn.Conv2d(input_nch, output_nch, kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(output_nch, output_nch, kernel_size, padding=padding)
        self.act = activation
        self.batch_norm = nn.BatchNorm2d(output_nch) if use_batchnorm else None

    def forward(self, x):
        x = self.conv0(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.act(x)
        x = self.conv1(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return self.act(x)


class UNet(nn.Module):
    def __init__(self, conv_channels, input_channels=3, output_channels=2, use_bn=True):
        super(UNet, self).__init__()
        self.n_stages = len(conv_channels)
        # define convolution blocks
        down_convs = []
        up_convs = []

        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        in_nch = input_channels
        for i, out_nch in enumerate(conv_channels):
            down_convs.append(UNetConvBlock(in_nch, out_nch, use_batchnorm=use_bn))
            up_conv_in_ch = 2 * out_nch if i < self.n_stages - 1 else out_nch # first up conv with equal channels
            up_conv_out_ch = out_nch if i == 0 else in_nch  # last up conv with channels equal to labels
            up_convs.insert(0, UNetConvBlock(up_conv_in_ch, up_conv_out_ch, use_batchnorm=use_bn))
            in_nch = out_nch

        self.down_convs = nn.ModuleList(down_convs)
        self.up_convs = nn.ModuleList(up_convs)

        # define output convolution
        self.out_conv = nn.Conv2d(conv_channels[0], output_channels, 1)

    def forward(self, x):
        # conv & downsampling
        down_sampled_fmaps = []
        for i in range(self.n_stages-1):
            x = self.down_convs[i](x)
            x = self.max_pooling(x)
            down_sampled_fmaps.insert(0, x)

        # center convs
        x = self.down_convs[self.n_stages-1](x)
        x = self.up_convs[0](x)

        # conv & upsampling
        for i, down_sampled_fmap in enumerate(down_sampled_fmaps):
            x = torch.cat([x, down_sampled_fmap], 1)
            x = self.up_convs[i+1](x)
            x = F.upsample(x, scale_factor=2, mode='bilinear')

        return self.out_conv(x)
        #x = self.out_conv(x)
        #return x if self.out_conv.out_channels == 1 else F.relu(x)


def get_train_val_files():
    SYN = SearchFile(SYNTHIA_CS)
    syn_imgs = SYN(datatype='image')[0:2975]
    syn_foreLabels = SYN(datatype='foregroundLabel')[0:2975]

    CS = SearchFile(CITYSCAPES)
    city_imgs = CS(datatype='image',
                                 pattern='train/*/*%s.%s' % (
                                     CS.postfix_map['image'], CS.data_format['image']))
    city_foreLabels = CS(datatype='foregroundLabel',
                                       pattern='train/*/*%s.%s' % (
                                       CS.postfix_map['foregroundLabel'], CS.data_format['foregroundLabel']))

    val_imgs = CS(datatype='image',
                   pattern='val/*/*%s.%s' % (
                       CS.postfix_map['image'], CS.data_format['image']))
    val_foreLabels = CS(datatype='foregroundLabel',
                         pattern='val/*/*%s.%s' % (
                             CS.postfix_map['foregroundLabel'], CS.data_format['foregroundLabel']))
    train_imgs = syn_imgs
    train_foreLabels = syn_foreLabels
    train_imgs = city_imgs
    train_foreLabels = city_foreLabels
    return train_imgs,train_foreLabels,val_imgs,val_foreLabels

def main():
    train_imgs, train_foreLabels, val_imgs, val_foreLabels=get_train_val_files()
    transform= transforms.Compose([RandomCrop(512), ToTensor()])

    TrainMixData = BaseDataset(train_imgs, train_foreLabels, transform)
    TestMixData = BaseDataset(val_imgs, val_foreLabels, ToTensor())
    sample = TrainMixData[0]

    model = UNet([32, 64, 128, 256, 512])


    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9, weight_decay=0.0005)
    # optimizer = optim.Adam(model.parameters(),lr=1e-4)
    train_loader = loader(TrainMixData, batch_size=10)
    criterion = CrossEntropyLoss2D()
    train(model,train_loader,optimizer,criterion,eval_func = lambda x,y:IOU(torch.argmax(x, dim=1),y),device='cuda')


if __name__ == '__main__':
    main()




