import os
from torch.utils.data import Dataset
import pandas as pd

import numpy as np
import glob
from PIL import Image

ROOT = "/blue/parisa.rashidi/subhashnerella/Datasets/"


class SampleFace(Dataset):
    def __init__(self, *args, **kwargs):
        self.data = None

    def _load(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

class DiscreteImages(SampleFace):
    def __init__(self, imgspath, n_aus, size=225):
        super().__init__()
        self.img_path = glob.glob(os.path.join(imgspath, "*"))
        self.n_aus = n_aus
        self.size = size
    
    def __len__(self):
        return len(self.img_path)
    
    def preprocess_image(self,image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((self.size, self.size))
        image = np.array(image).astype(np.uint8)
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, idx):
        sample = dict()
        sample["image"] = self.preprocess_image(self.img_path[idx])
        sample["aus"] = np.random.randint(2,size=self.n_aus)
        return sample 


