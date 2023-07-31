import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

def _make_block(num_layers, num_channels, kernel_sizes):
    layers = []
    for i in range(num_layers):
        layers.append(BasicBlock(num_channels[i], num_channels[i+1], kernel_sizes[i]))
    
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    return nn.Sequential(*layers)

class CNets(nn.Module):
    def __init__(self, in_channels, out_channels, num_para):
        super().__init__()

        self.block1 = _make_block(num_layers=3, num_channels=(in_channels,out_channels*3,out_channels*2,out_channels), kernel_sizes=(5,1,1))
        self.block2 = _make_block(num_layers=3, num_channels=(out_channels,out_channels,out_channels,out_channels), kernel_sizes=(3,1,1))
        self.block3 = _make_block(num_layers=3, num_channels=(out_channels,out_channels,out_channels,out_channels), kernel_sizes=(3,1,1))
        self.block4 = _make_block(num_layers=3, num_channels=(out_channels,out_channels,out_channels,out_channels), kernel_sizes=(2,1,1))
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, num_para)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = torch.flatten(out,1)
        out = self.fc2(out)

        return out



class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super().__init__()

        self.block1 = _make_block(num_layers=2, num_channels=(in_channels,out_channels,out_channels), kernel_sizes=(5,5))
        self.block2 = _make_block(num_layers=2, num_channels=(out_channels,out_channels*2,out_channels*2), kernel_sizes=(3,3))
        self.fc1 = nn.Linear(out_channels, out_channels*8)
        self.fc2 = nn.Linear(out_channels*8, num_class)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = torch.flatten(out,1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out
