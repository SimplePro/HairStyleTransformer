from torchvision.models import ResNet50_Weights, resnet50
import torch.nn as nn
import torch

class SSL(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

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

            nn.Softmax(dim=1)
        ) # output shape: (n_classes)


    def forward(self, x):
        features = self.features(x)
        out = self.fc(features)
        return out
    

if __name__ == '__main__':
    print(SSL(n_classes=2)(torch.randn(1, 3, 256, 256)).shape)