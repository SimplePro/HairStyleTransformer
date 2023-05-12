import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50, ResNet101_Weights, resnet101
from torch.nn.init import xavier_normal_
from torchsummary import summary


"""
Referenced from https://github.com/Lextal/pspnet-pytorch/blob/master/pspnet.py
"""


class ResNet101Extractor(nn.Module):
    def __init__(self):
        super(ResNet101Extractor, self).__init__()
        model = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:7])
    def forward(self, x):
        return self.features(x)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        pyramid_levels = len(sizes)
        out_channels = in_channels // pyramid_levels

        pooling_layers = nn.ModuleList()
        for size in sizes:
            layers = [nn.AdaptiveAvgPool2d(size), nn.Conv2d(in_channels, out_channels, kernel_size=1)]
            pyramid_layer = nn.Sequential(*layers)
            pooling_layers.append(pyramid_layer)

        self.pooling_layers = pooling_layers

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        features = [x]
        for pooling_layer in self.pooling_layers:
            # pool with different sizes
            pooled = pooling_layer(x)

            # upsample to original size
            upsampled = F.upsample(pooled, size=(h, w), mode='bilinear')

            features.append(upsampled)

        return torch.cat(features, dim=1)


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size=None):
        super().__init__()
        self.upsample_size = upsample_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = 2 * x.size(2), 2 * x.size(3)
        f = F.upsample(x, size=size, mode='bilinear')
        return self.conv(f)


class PSPNet(nn.Module):
    def __init__(self, num_class=1, sizes=(1, 2, 3, 6)):
        super(PSPNet, self).__init__()

        self.base_network = ResNet101Extractor()
        feature_dim = 1024
        
        self.psp = PyramidPoolingModule(in_channels=feature_dim, sizes=sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = UpsampleLayer(2*feature_dim, 256)
        self.up_2 = UpsampleLayer(256, 64)
        self.up_3 = UpsampleLayer(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, num_class, kernel_size=1)
        )

        self._init_weight()

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        f = self.base_network(x)
        p = self.psp(f)
        p = self.drop_1(p)
        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)

        if (p.size(2) != h) or (p.size(3) != w):
            p = F.interpolate(p, size=(h, w), mode='bilinear')

        p = self.drop_2(p)

        return self.final(p)

    def _init_weight(self):
        layers = [self.up_1, self.up_2, self.up_3, self.final]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                xavier_normal_(layer.weight.data)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)


class DownSampling(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.normal_(1.0, 0.02)
        self.bn.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x += self.bias.reshape(1, self.bias.size()[0], 1, 1)
        x = self.bn(x)
        x = F.gelu(x)

        return x


class SSL(nn.Module):

    def __init__(self, n_classes, temperature=1, state_dict_path=""):
        super().__init__()

        self.n_classes = n_classes
        self.temperature = temperature

        self.last_layer = nn.ModuleList([])

        if state_dict_path == "":
            # input_shape: (3, 256, 256)
            self.features = nn.Sequential(
                *list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1],
            ) # feature shape: (2048, 1, 1)

            feature_dim = 2048

        else:
            pspnet = PSPNet(num_class=1)
            pspnet.load_state_dict(torch.load(state_dict_path)["weight"])

            self.features = pspnet.base_network # (1024, 16, 16)

            self.last_layer.add_module(
                "down_sampling0",
                DownSampling(in_channels=1024, out_channels=256)
            ) # (256, 8, 8)

            self.last_layer.add_module("dropout0", nn.Dropout(0.3))

            self.last_layer.add_module(
                "down_sampling1",
                DownSampling(in_channels=256, out_channels=256)
            ) # (256, 4, 4)

            self.last_layer.add_module("dropout1", nn.Dropout(0.3))

            self.last_layer.add_module(
                "down_sampling2",
                DownSampling(in_channels=256, out_channels=256, kernel_size=4, stride=1, padding=0)
            ) # (256, 1, 1)

            self.last_layer.add_module("dropout2", nn.Dropout(0.3))

            feature_dim = 256


        self.last_layer.add_module(
            "last_layer",
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_dim, 256),
                nn.Dropout(0.3),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.Dropout(0.3),
                nn.GELU(),
                nn.Linear(256, n_classes),
                nn.GELU(),
            ) # output shape: (n_classes)
        )


    def forward(self, x):
        features = self.features(x)
        out = features

        for module in self.last_layer:
            out = module(out)

        out /= self.temperature
        out = F.softmax(out, dim=1)
        return out, features



if __name__ == '__main__':
    summary(SSL(n_classes=2, state_dict_path="pspnet_resnet101.pth").cuda().features, (3, 256, 256))