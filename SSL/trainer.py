import torch

from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from glob import glob

from tqdm import tqdm

from models import SSL

import os

from random import choices


class SSLTrainer:

    def __init__(
        self,
        device,
        labeled_dir,
        unlabeled_dir,
        weak_transform,
        strong_transform,
        batch_size,
        threshold=0.9,
        temperature=0.95,
        unlabeled_lambda=1
    ):
        
        self.device = device
        
        self.labeled_dataset = ImageFolder(labeled_dir,
                                           transform=transforms.Compose([
                                                transforms.Resize((256, 256)),
                                                transforms.ToTensor()
                                           ]))
        self.unlabeled_dataset = ImageFolder(unlabeled_dir,
                                             transform=transforms.Compose([
                                                transforms.Resize((256, 256)),
                                                transforms.ToTensor()
                                           ]))

        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        self.batch_size = batch_size

        self.n_classes = len(os.listdir(labeled_dir))

        self.ssl = SSL(n_classes=self.n_classes, temperature=temperature).to(self.device)

        self.optim = Adam([
            {'params': self.ssl.features.parameters(), 'lr': 1e-4},
            {'params': self.ssl.fc.parameters(), 'lr': 1e-3}
        ], betas=(0.5, 0.999))

        self.threshold = threshold
        self.T = temperature
        self.unlabeled_lambda = unlabeled_lambda

    def get_batch(self, dataset, batch_size):
        index_ = list(range(len(dataset)))
        index_ = choices(index_, k=batch_size)

        x = torch.zeros(batch_size, 3, 256, 256)
        y = torch.zeros(batch_size).type(torch.long)

        for i, index in enumerate(index_):
            x[i], y[i] = dataset.__getitem__(index)
        
        return x, y
    
    def train(self, max_iter, log_iter):

        labeled_history = []
        unlabeled_history = []

        cur_iter = 0

        for cur_iter in tqdm(range(max_iter)):
            labeled_x, labeled_y = self.get_batch(self.labeled_dataset, self.batch_size)
            labeled_x, labeled_y = labeled_x.to(self.device), labeled_y.to(self.device)
            
            unlabeled_x, _ = self.get_batch(self.unlabeled_dataset, self.batch_size)
            unlabeled_x = unlabeled_x.to(self.device)

            labeled_pred = self.ssl(labeled_x)
            labeled_loss = F.cross_entropy(labeled_pred, labeled_y)

            unlabeled_weak_pred = self.ssl(self.weak_transform(unlabeled_x))
            unlabeled_strong_pred = self.ssl(self.strong_transform(unlabeled_x))

            unlabeled_loss = F.cross_entropy(
                unlabeled_strong_pred,
                torch.argmax(unlabeled_weak_pred, dim=1).type(torch.long),
                reduction="none"
            )

            unlabeled_loss = torch.mean(
                (torch.max(unlabeled_weak_pred, dim=1).values > self.threshold) * unlabeled_loss
            )

            loss = labeled_loss + self.unlabeled_lambda * unlabeled_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            labeled_history.append(labeled_loss.item())
            unlabeled_history.append(unlabeled_loss.item())
            
            if (cur_iter+1) % log_iter == 0: 
                print(f"iter: {cur_iter+1}/{max_iter}, labeled_loss: %.4f, unlabeled_loss: %.4f"
                      % (sum(labeled_history)/len(labeled_history), sum(unlabeled_history)/len(unlabeled_history)),
                      end="\n\n")


if __name__ == '__main__':
    weak_transform = transforms.Compose([
        transforms.RandomCrop(size=(192, 192)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((256, 256))
    ])

    strong_transform = transforms.Compose([
        transforms.RandomCrop(size=(192, 192)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomInvert(p=0.5),
        transforms.Resize((256, 256))
    ])

    ssl_trainer = SSLTrainer(
        device=torch.device("cuda"),
        labeled_dir="ssl_dataset/forehead/labeled",
        unlabeled_dir="ssl_dataset/forehead/unlabeled",
        weak_transform=weak_transform,
        strong_transform=strong_transform,
        batch_size=64,
        threshold=0.9,
        temperature=0.95,
        unlabeled_lambda=1
    )

    ssl_trainer.train(max_iter=1000, log_iter=10)

    