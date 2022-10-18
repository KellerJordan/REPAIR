import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LayerNorm2d(nn.Module):
    def __init__(self, nchan):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(nchan))
        self.bias = nn.Parameter(torch.zeros(nchan))
    def forward(self, x):
        x = x - x.mean(1, keepdim=True)
        x = x / x.std(1, keepdim=True, unbiased=False)
        x = x * self.weight.reshape(1, -1, 1, 1)
        x = x + self.bias.reshape(1, -1, 1, 1)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = LayerNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = LayerNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True),
                LayerNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, w=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = w*16

        self.conv1 = nn.Conv2d(3, w*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = LayerNorm2d(w*16)
        self.layer1 = self._make_layer(block, w*16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, w*32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, w*64, num_blocks[2], stride=2)
        self.linear = nn.Linear(w*64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(w=1):
    return ResNet(BasicBlock, [3, 3, 3], w=w)

