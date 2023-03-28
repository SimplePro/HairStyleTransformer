import torch
from torch import nn
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from glob import glob

from tqdm import tqdm

from models import SSL

import os


class SSLTrainer:

    def __init__(
        self,
        device,
        labeled_dir,
        unlabeled_dir,
        weak_transform,
        strong_transform,
        batch_size,
    ):
        
        self.device = device
        
        self.labeled_dataset = ImageFolder(labeled_dir, transform=weak_transform)
        self.unlabeled_dataset = ImageFolder(unlabeled_dir, transform=strong_transform)

        self.labeled_loader = DataLoader(self.labeled_dataset, batch_size=batch_size)
        self.unlabeled_loader = DataLoader(self.unlabeled_dataset, batch_size=batch_size)

        self.n_classes = len(os.listdir(labeled_dir))

        self.ssl = SSL(n_classes=self.n_classes).to(self.device)

        self.optim = Adam([
            {'params': self.ssl.features.parameters(), 'lr': 1e-5},
            {'params': self.ssl.fc.parameters(), 'lr': 1e-4}
        ], betas=(0.5, 0.999))

        self.criterion = nn.CrossEntropyLoss()

    
    def train_epoch():

        avg_loss = 0

        pass