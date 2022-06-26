import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['SENet', 'senet50']

from backbones.countFLOPS import count_model_flops
from backbones.utils import _calc_width


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# This SEModule is not used.
class SEModule(nn.Module):

    def __init__(self, planes, compress_rate):
        super(SEModule, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes // compress_rate, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(planes // compress_rate, planes, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = F.avg_pool2d(module_input, kernel_size=module_input.size(2))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # SENet
        compress_rate = 16
        # self.se_block = SEModule(planes * 4, compress_rate)  # this is not used.
        self.conv4 = nn.Conv2d(planes * 4, planes * 4 // compress_rate, kernel_size=1, stride=1, bias=True)
        self.conv5 = nn.Conv2d(planes * 4 // compress_rate, planes * 4, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        ## senet
        out2 = F.avg_pool2d(out, kernel_size=out.size(2))
        out2 = self.conv4(out2)
        out2 = self.relu(out2)
        out2 = self.conv5(out2)
        out2 = self.sigmoid(out2)
        # out2 = self.se_block.forward(out)  # not used

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out2 * out + residual
        # out = out2 + residual  # not used
        out = self.relu(out)
        return out


class SENet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=True):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        if not self.include_top:
            return x

        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x

class sphere64(nn.Module):
    def __init__(self,classnum=10574,feature=False):
        super(sphere64, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)

        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)



        self.conv1_4 = nn.Conv2d(64,64,3,1,1)
        self.relu1_4 = nn.PReLU(64)
        self.conv1_5 = nn.Conv2d(64,64,3,1,1)
        self.relu1_5 = nn.PReLU(64)

        self.conv1_6 = nn.Conv2d(64,64,3,1,1)
        self.relu1_6 = nn.PReLU(64)
        self.conv1_7 = nn.Conv2d(64,64,3,1,1)
        self.relu1_7 = nn.PReLU(64)


        self.conv1_8 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_8 = nn.PReLU(64)
        self.conv1_9 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_9 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)

        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)

        self.conv2_6 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_6 = nn.PReLU(128)
        self.conv2_7 = nn.Conv2d(128,128,3,1,1)
        self.relu2_7 = nn.PReLU(128)

        self.conv2_8 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_8 = nn.PReLU(128)
        self.conv2_9 = nn.Conv2d(128,128,3,1,1)
        self.relu2_9 = nn.PReLU(128)

        self.conv2_10 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_10 = nn.PReLU(128)
        self.conv2_11 = nn.Conv2d(128,128,3,1,1)
        self.relu2_11 = nn.PReLU(128)

        self.conv2_12 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_12 = nn.PReLU(128)
        self.conv2_13 = nn.Conv2d(128,128,3,1,1)
        self.relu2_13 = nn.PReLU(128)

        self.conv2_14 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_14 = nn.PReLU(128)
        self.conv2_15 = nn.Conv2d(128,128,3,1,1)
        self.relu2_15 = nn.PReLU(128)

        self.conv2_16 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_16 = nn.PReLU(128)
        self.conv2_17 = nn.Conv2d(128,128,3,1,1)
        self.relu2_17 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)

        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)


        self.conv3_10 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_10 = nn.PReLU(256)

        self.conv3_11 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_11 = nn.PReLU(256)
        self.conv3_12 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_12 = nn.PReLU(256)

        self.conv3_13 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_13 = nn.PReLU(256)
        self.conv3_14 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_14 = nn.PReLU(256)

        self.conv3_15 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_15 = nn.PReLU(256)
        self.conv3_16 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_16 = nn.PReLU(256)

        self.conv3_17 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_17 = nn.PReLU(256)
        self.conv3_18 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_18 = nn.PReLU(256)

        self.conv3_19 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_19 = nn.PReLU(256)
        self.conv3_20 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_20 = nn.PReLU(256)

        self.conv3_21 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_21 = nn.PReLU(256)
        self.conv3_22 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_22 = nn.PReLU(256)

        self.conv3_23 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_23 = nn.PReLU(256)
        self.conv3_24 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_24 = nn.PReLU(256)

        self.conv3_25 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_25 = nn.PReLU(256)
        self.conv3_26 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_26 = nn.PReLU(256)

        self.conv3_27 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_27 = nn.PReLU(256)
        self.conv3_28 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_28 = nn.PReLU(256)

        self.conv3_29 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_29 = nn.PReLU(256)
        self.conv3_30 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_30 = nn.PReLU(256)

        self.conv3_31 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_31 = nn.PReLU(256)
        self.conv3_32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_32 = nn.PReLU(256)
        self.conv3_33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_33 = nn.PReLU(256)


        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.conv4_4 = nn.Conv2d(512,512,3,1,1)
        self.relu4_4 = nn.PReLU(512)
        self.conv4_5 = nn.Conv2d(512,512,3,1,1)
        self.relu4_5 = nn.PReLU(512)

        self.conv4_6 = nn.Conv2d(512,512,3,1,1)
        self.relu4_6 = nn.PReLU(512)
        self.conv4_7 = nn.Conv2d(512,512,3,1,1)
        self.relu4_7 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
        x = x + self.relu1_5(self.conv1_5(self.relu1_4(self.conv1_4(x))))
        x = x + self.relu1_7(self.conv1_7(self.relu1_6(self.conv1_6(x))))


        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))
        x = x + self.relu2_7(self.conv2_7(self.relu2_6(self.conv2_6(x))))
        x = x + self.relu2_9(self.conv2_9(self.relu2_8(self.conv2_8(x))))
        x = x + self.relu2_11(self.conv2_11(self.relu2_10(self.conv2_10(x))))
        x = x + self.relu2_13(self.conv2_13(self.relu2_12(self.conv2_12(x))))
        x = x + self.relu2_15(self.conv2_15(self.relu2_14(self.conv2_14(x))))
        x = x + self.relu2_17(self.conv2_17(self.relu2_16(self.conv2_16(x))))



        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = x + self.relu3_11(self.conv3_11(self.relu3_10(self.conv3_10(x))))
        x = x + self.relu3_13(self.conv3_13(self.relu3_12(self.conv3_12(x))))
        x = x + self.relu3_15(self.conv3_15(self.relu3_14(self.conv3_14(x))))
        x = x + self.relu3_17(self.conv3_17(self.relu3_16(self.conv3_16(x))))

        x = x + self.relu3_19(self.conv3_19(self.relu3_18(self.conv3_18(x))))
        x = x + self.relu3_21(self.conv3_21(self.relu3_20(self.conv3_20(x))))
        x = x + self.relu3_23(self.conv3_23(self.relu3_22(self.conv3_22(x))))
        x = x + self.relu3_25(self.conv3_25(self.relu3_24(self.conv3_24(x))))

        x = x + self.relu3_27(self.conv3_27(self.relu3_26(self.conv3_26(x))))
        x = x + self.relu3_29(self.conv3_29(self.relu3_28(self.conv3_28(x))))
        x = x + self.relu3_31(self.conv3_31(self.relu3_30(self.conv3_20(x))))
        x = x + self.relu3_33(self.conv3_33(self.relu3_32(self.conv3_32(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        x = x + self.relu4_5(self.conv4_5(self.relu4_4(self.conv4_4(x))))
        x = x + self.relu4_7(self.conv4_7(self.relu4_7(self.conv4_6(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        if self.feature: return x

        return x
def senet50(**kwargs):
    """Constructs a SENet-50 model.
    """
    model = SENet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
def _test():
    import torch

    pretrained = False

    models = [
        senet50
    ]

    for model in models:

        net = model()
        #print(net)
        # net.train()
        net.eval()
        x = torch.randn(1, 3, 224, 224)

        y = net(x)
        y.sum().backward()
        print(y.size())

        assert (tuple(y.size()) == (1, 2048,1,1))
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        flops = count_model_flops(net,input_res=[224,224])
        print("m={}, {}".format(model.__name__, flops))




if __name__ == "__main__":
    _test()
