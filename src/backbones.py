import torch
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, resnet18

from functools import reduce
from operator import __add__

from torch.distributions import Bernoulli
import math

# First model: Efficientnet v2

def effinet(variant='s'):
    if variant == 's':
        net = efficientnet_v2_s(weights='DEFAULT')
    elif variant == 'm':
        net = efficientnet_v2_m(weights='DEFAULT')
    elif variant == 'l':
        net = efficientnet_v2_l(weights='DEFAULT')
    else:
        raise ValueError(
            'the version of the effinet you\'re looking for is not found!'
        )
    
    net.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Flatten()
    )
    return net


# Second model: Resnet12

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

class DropBlock(torch.nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        #self.gamma = gamma
        #self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)))
            #print((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            #print (block_mask.size())
            #print (x.size())
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = torch.nn.BatchNorm2d(planes)
        self.maxpool = torch.nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = torch.nn.functional.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class ResNet12(torch.nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, flatten=False):
        self.inplanes = 3
        super(ResNet12, self).__init__()
        self.final_feat_dim = 640
        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = torch.nn.Sequential(
                  torch.nn.AvgPool2d(5, stride=1),
                  Flatten()
            )
        self.keep_prob = keep_prob
        self.flatten = flatten
        self.keep_avg_pool = avg_pool
        self.dropout = torch.nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        return x

def resnet(variant='18'):
    if variant=='18':
        net = resnet18(pretrained=True)
        net.fc = torch.nn.Flatten()
    elif variant=='12':
        net = ResNet12(BasicBlock, keep_prob=1.0, drop_rate=0.1, dropblock_size=2, avg_pool=False, flatten=True)
    else:
        raise ValueError(
            'the version of the resnet you\'re looking for is not found!'
        )
    
    return net


# Third model: ConvNet

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, torch.nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, torch.nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Conv2d_fw(torch.nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = torch.nn.functional.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = torch.nn.functional.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out

class BatchNorm2d_fw(torch.nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1])
        running_var = torch.ones(x.data.size()[1])
        if self.weight.fast is not None and self.bias.fast is not None:
            out = torch.nn.functional.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = torch.nn.functional.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

class ConvBlock(torch.nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = torch.nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = torch.nn.BatchNorm2d(outdim)
        self.relu   = torch.nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = torch.nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = torch.nn.Sequential(*self.parametrized_layers)

    def forward(self,x):
        out = self.trunk(x)
        return out


class ConvNet(torch.nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for misal:4 layers
            trunk.append(B)

        if flatten:
            trunk.append(torch.nn.Dropout(0.5))
            trunk.append(Flatten())

        self.trunk = torch.nn.Sequential(*trunk)
        self.final_feat_dim = 640

    def forward(self,x):
        out = self.trunk(x)
        return out