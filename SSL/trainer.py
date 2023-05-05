# FixMatch Model Trainer

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
        labeled_dir, # labeled_dataset의 directory path
        unlabeled_dir, # labeled_dataset의 directory path
        weak_transform, # 약한 augmentation에 사용될 transform
        strong_transform, # 강한 augmentation에 사용될 transform
        batch_size,
        threshold=0.9, # FixMatch 모델을 학습할 때 쓰일 threshold
        temperature=0.95, # FixMatch 모델을 학습할 때 쓰일 temperature
        unlabeled_lambda=1 # unsupervised_loss에 곱해질 lambda constant
    ):
        
        self.device = device

        # labeled_dataset 불러오기
        self.labeled_dataset = ImageFolder(labeled_dir,
                                           transform=transforms.Compose([
                                                transforms.Resize((256, 256)),
                                                transforms.ToTensor()
                                           ]))

        # unlabeled_dataset 불러오기
        self.unlabeled_dataset = ImageFolder(unlabeled_dir,
                                             transform=transforms.Compose([
                                                transforms.Resize((256, 256)),
                                                transforms.ToTensor()
                                           ]))

        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        self.batch_size = batch_size

        # class의 개수
        self.n_classes = len(os.listdir(labeled_dir))

        # FixMatch 모델
        self.ssl = SSL(n_classes=self.n_classes, temperature=temperature).to(self.device)
        
        # pre-trained된 features모델과 새로 학습시키는 fc 레이어의 옵션을 다르게 줌.
        self.optim = Adam([
            {'params': self.ssl.features.parameters(), 'lr': 1e-4},
            {'params': self.ssl.fc.parameters(), 'lr': 1e-3}
        ], betas=(0.5, 0.999))

        self.threshold = threshold
        self.T = temperature
        self.unlabeled_lambda = unlabeled_lambda

    def get_batch(self, dataset, batch_size):
        # 랜덤한 index 추출
        index_ = choices(range(len(dataset)), k=batch_size)

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
            # mini batch 가져오기
            labeled_x, labeled_y = self.get_batch(self.labeled_dataset, self.batch_size)
            labeled_x, labeled_y = labeled_x.to(self.device), labeled_y.to(self.device)
            
            unlabeled_x, _ = self.get_batch(self.unlabeled_dataset, self.batch_size)
            unlabeled_x = unlabeled_x.to(self.device)

            # labeled_x에 대한 class 예측
            labeled_pred = self.ssl(labeled_x)
            labeled_loss = F.cross_entropy(labeled_pred, labeled_y) # supervised-loss 계산

            # 약한 augmentation이 적용된 unlabeled_x에 대한 class 예측
            unlabeled_weak_pred = self.ssl(self.weak_transform(unlabeled_x))

            # 강한 augmentation이 적용된 unlabeled_x에 대한 class 예측
            unlabeled_strong_pred = self.ssl(self.strong_transform(unlabeled_x))

            # unsupervised-loss 계산
            unlabeled_loss = F.cross_entropy(
                unlabeled_strong_pred, # input: 강한 augmentation이 적용된 unlabeled_x에 대한 class
                torch.argmax(unlabeled_weak_pred, dim=1).type(torch.long), # target: unlabeled_weak_pred의 psuedo_label
                reduction="none"
            )

            condition = (torch.max(unlabeled_weak_pred, dim=1).values > self.threshold)
            unlabeled_loss = condition * unlabeled_loss # unlabeled_weak_pred의 confidence가 threshold보다 작으면 loss 계산안됨
            unlabeled_loss /= max(condition.sum(), 1) # unsupervised-loss 계산에 참여하는 element의 개수로 나눔. (평균 구하기)

            loss = labeled_loss + self.unlabeled_lambda * unlabeled_loss # 전체 loss 계산

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

    