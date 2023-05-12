# FixMatch Model Trainer

import torch

from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

from glob import glob

from tqdm import tqdm

from models import SSL

import os

from random import choices

from utils import SSLDataset


class SSLTrainer:

    def __init__(
        self,
        device,
        labeled_dir, # labeled_dataset의 directory path
        unlabeled_dir, # labeled_dataset의 directory path
        labeled_transform, # 라벨링된 데이터셋에 사용될 transform
        weak_transform, # 약한 augmentation에 사용될 transform
        strong_transform, # 강한 augmentation에 사용될 transform
        batch_size,
        threshold=0.9, # FixMatch 모델을 학습할 때 쓰일 threshold
        temperature=0.95, # FixMatch 모델을 학습할 때 쓰일 temperature
        unlabeled_lambda_opt=(0.05, 1, 0.05) # unsupervised_loss에 곱해질 lambda constant의 시작, 끝, 증가간격
    ):
        
        self.device = device

        # labeled_dataset 불러오기
        self.labeled_dataset = SSLDataset(
            dir_=labeled_dir,
            transform_=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]))

        # unlabeled_dataset 불러오기
        self.unlabeled_dataset = ImageFolder(
            unlabeled_dir,
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]))

        self.labeled_transform = labeled_transform
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        self.batch_size = batch_size

        # class의 개수
        self.n_classes = len(os.listdir(labeled_dir))

        # FixMatch 모델
        self.ssl = SSL(n_classes=self.n_classes, temperature=temperature, state_dict_path="pspnet_resnet101.pth").to(self.device)
        
        # pre-trained된 features모델과 새로 학습시키는 last_layer의 옵션을 다르게 줌.
        self.optim = Adam([
            {'params': self.ssl.features.parameters(), 'lr': 2e-4},
            {'params': self.ssl.last_layer.parameters(), 'lr': 1e-3}
        ], betas=(0.5, 0.999))

        self.threshold: float = threshold
        self.T: float = temperature

        self.unlabeled_lambda_opt: tuple = unlabeled_lambda_opt
        self.unlabeled_lambda: float = unlabeled_lambda_opt[0]

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

            labeled_x = self.labeled_transform(labeled_x)
            
            unlabeled_x, _ = self.get_batch(self.unlabeled_dataset, self.batch_size)
            unlabeled_x = unlabeled_x.to(self.device)

            # labeled_x에 대한 class 예측
            labeled_pred, _ = self.ssl(labeled_x)
            labeled_loss = F.cross_entropy(labeled_pred, labeled_y) # supervised-loss 계산

            # 약한 augmentation이 적용된 unlabeled_x에 대한 class 예측
            unlabeled_weak_x = self.weak_transform(unlabeled_x)
            unlabeled_weak_pred, features = self.ssl(unlabeled_weak_x)

            # unlabeled_weak_images = make_grid(unlabeled_weak_x, nrow=4)
            # TF.to_pil_image(unlabeled_weak_images.cpu().detach()).save("./unlabeled_weak_images.png")

            # features = make_grid(torch.mean(features, dim=1, keepdim=True), nrow=4)
            # print(features.shape)
            # TF.to_pil_image(features.cpu().detach()).save("./features.png")

            # 강한 augmentation이 적용된 unlabeled_x에 대한 class 예측
            unlabeled_strong_x = self.strong_transform(unlabeled_x)
            unlabeled_strong_pred, _ = self.ssl(unlabeled_strong_x)

            # unlabeled_strong_images = make_grid(unlabeled_strong_x, nrow=4)
            # TF.to_pil_image(unlabeled_strong_images.cpu().detach()).save("./unlabeled_strong_images.png")

            # unsupervised-loss 계산
            unlabeled_loss = F.cross_entropy(
                unlabeled_strong_pred, # input: 강한 augmentation이 적용된 unlabeled_x에 대한 class
                torch.argmax(unlabeled_weak_pred, dim=1).type(torch.long), # target: unlabeled_weak_pred의 psuedo_label
                reduction="none"
            )

            condition = (torch.max(unlabeled_weak_pred, dim=1).values > self.threshold)
            unlabeled_loss = condition * unlabeled_loss # unlabeled_weak_pred의 confidence가 threshold보다 작으면 loss 계산안됨
            unlabeled_loss /= max(condition.sum(), 1) # unsupervised-loss 계산에 참여하는 element의 개수로 나눔. (평균 구하기)
            unlabeled_loss = sum(unlabeled_loss)

            loss = labeled_loss + self.unlabeled_lambda * unlabeled_loss # 전체 loss 계산

            # unlabeled_lambda 증가 (만약 self.unlabeled_lambda_opt[1] 보다 작다면)
            # unlabeled_lambda_opt = (start_lambda, end_lambda, step)
            self.unlabeled_lambda = min(
                self.unlabeled_lambda+self.unlabeled_lambda_opt[2],
                self.unlabeled_lambda_opt[1]
            )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            labeled_history.append(labeled_loss.item())
            unlabeled_history.append(unlabeled_loss.item())
            
            if (cur_iter+1) % log_iter == 0:
                print(f"iter: {cur_iter+1}/{max_iter}, labeled_loss: %.4f, unlabeled_loss: %.4f"
                      % (sum(labeled_history)/len(labeled_history), sum(unlabeled_history)/len(unlabeled_history)),
                    #   end="\n\n"
                )
                print(torch.max(labeled_pred, dim=1)[1], torch.max(unlabeled_weak_pred, dim=1)[1])
                
                for i in range(len(labeled_x)):
                    TF.to_pil_image(labeled_x[i].cpu()).save(f"./labeled_x{i}.jpg")

                for i in range(len(unlabeled_x)):
                    TF.to_pil_image(self.weak_transform(unlabeled_x[i].cpu())).save(f"./unlabeled_x{i}.jpg")



if __name__ == '__main__':
    labeled_transform = transforms.Compose([
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((256, 256))
    ])

    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    strong_transform = transforms.Compose([
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.Resize((256, 256))
    ])

    ssl_trainer = SSLTrainer(
        device=torch.device("cuda"),
        labeled_dir="ssl_dataset/forehead/labeled",
        unlabeled_dir="ssl_dataset/forehead/unlabeled",
        labeled_transform=labeled_transform,
        weak_transform=weak_transform,
        strong_transform=strong_transform,
        batch_size=32,
        threshold=0.9,
        temperature=0.95,
        unlabeled_lambda_opt=(0.05, 1, 0.05)
    )

    ssl_trainer.ssl.load_state_dict(torch.load("ssl1.pth"))

    ssl_trainer.train(max_iter=1000, log_iter=10)

    # torch.save(ssl_trainer.ssl.state_dict(), "./ssl1.pth")
