import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torchvision.models import ResNet50_Weights, resnet50


class SSL(nn.Module):

    def __init__(self, n_classes, temperature=1):
        super().__init__()

        self.n_classes = n_classes
        self.temperature = temperature

        # input_shape: (3, 256, 256)
        self.features = nn.Sequential(
            *list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]
        ) # feature shape: (2048, 1, 1)

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(2048, 100),
            nn.GELU(),

            nn.Linear(100, n_classes),
            nn.GELU(),
        ) # output shape: (n_classes)


    def forward(self, x):
        features = self.features(x)
        out = self.fc(features)
        out /= self.temperature
        out = softmax(out, dim=1)
        return out
    

if __name__ == '__main__':
    print(SSL(n_classes=2)(torch.randn(1, 3, 256, 256)).shape)