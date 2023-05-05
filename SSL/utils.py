# label_frame.csv 를 기반으로 ssl_dataset을 추가하는 utils file

import pandas as pd
import numpy as np

from PIL import Image

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import os

from random import shuffle

from glob import glob


class SSLDataset(Dataset):

    def __init__(self,
                 dir_, # dataset directory
                 transform_ # transform
    ):
        self.dir_ = dir_
        self.classes = os.listdir(self.dir_)
        self.transform_ = transform_

        self.selected_data = []
        self.unselected_data = []

        # class별로 데이터의 개수를 확인하고, 가장 작은 값을 대입
        # -> 데이터 편향을 아예 없애기 위함임.
        self.data_len_per_class = min([len(os.listdir(os.path.join(self.dir, cls))) for cls in self.classes])

        # class를 돌면서
        for cls in self.classes:
            paths = glob(os.path.join(self.dir, cls)) # 각 클래스의 이미지 경로들을
            shuffle(paths) # 섞어줌

            for i in range(len(paths)):

                if i < self.data_len_per_class: # 편향되지 않게 데이터 개수를 고려하여 selected_data 변수에 추가하고
                    self.selected_data.append((paths[i], cls))

                else: # 그 외는 사용하지 않는 데이터 변수에 추가함
                    self.unselected_data.append((paths[i], cls))

    def __len__(self):
        return len(self.selected_paths)
        
    def __getitem__(self, index):
        path, y = self.selected_data[index]
        x = Image.open(path)
        x = self.transform_(x)
        return (x, y)
                



if __name__ == '__main__':
    # label_frame.csv 읽기
    label_frames = pd.read_csv("../dataset/label_frame.csv")
    label_array = label_frames.to_numpy().tolist()

    # hair_img를 label에 따라 분류하여 ssl_dataset에 추가
    for file, forehead, length in tqdm(label_array):
        # hair_img 읽기
        hair_img = Image.open(f"../dataset/hair/{file}")

        # labeled data라면
        if forehead in ["0", "1", "2"]:
            hair_img.save(f"ssl_dataset/forehead/labeled/{forehead}/{file}") # 저장

        # unlabeled data라면
        elif type(forehead) == float and np.isnan(forehead):
            hair_img.save(f"ssl_dataset/forehead/unlabeled/{file}")

        # label이 -1인 datasample은 아예 사용하지 않음