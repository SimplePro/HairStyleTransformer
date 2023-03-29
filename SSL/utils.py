import pandas as pd
import numpy as np

from PIL import Image

from tqdm import tqdm


if __name__ == '__main__':
    label_frames = pd.read_csv("../dataset/label_frame.csv")
    label_array = label_frames.to_numpy().tolist()

    for file, forehead, length in tqdm(label_array):
        hair_img = Image.open(f"../dataset/hair/{file}")

        if forehead in ["0", "1", "2"]:
            hair_img.save(f"ssl_dataset/forehead/labeled/{forehead}/{file}")

        elif type(forehead) == float and np.isnan(forehead):
            hair_img.save(f"ssl_dataset/forehead/unlabeled/{file}")