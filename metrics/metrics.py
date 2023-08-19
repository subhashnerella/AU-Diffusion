from torch_fidelity import calculate_metrics
from einops import rearrange
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
from collections import defaultdict
import json
import torch
import numpy as np

class Faces(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = None
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = Image.open(sample)
        sample = torch.tensor(np.array(sample))
        sample = rearrange(sample,'h w c -> c h w')
        return sample
    
#'logs/2023-07-14T21-12-33_AU-ldm-vq/ImagesforMetrics'
class InputDataset(Faces):
    def __init__(self, epoch, path, split='train'):
        super().__init__()
        epoch = 'epoch'+str(epoch).zfill(3)
        path = os.path.join(path,split,epoch)
        self.data = glob.glob(os.path.join(path,'inputs*.png'))
        self.data.sort()


class GeneratedDataset(Faces):
    def __init__(self, epoch, path, split='train') -> None:
        super().__init__()
        epoch = 'epoch'+str(epoch).zfill(3)
        path = os.path.join(path,split,epoch)
        self.data = glob.glob(os.path.join(path,'img2img*.png'))
        self.data.sort()

def main():
    metrics_dict = defaultdict(dict)
    for epoch in range(11):
        print('##########################################################################################################################################################################')
        print(epoch)
        path = 'logs/2023-07-14T21-12-33_AU-ldm-vq/ImagesforMetrics'
        input1_train = GeneratedDataset(epoch,path)
        input2_train = InputDataset(epoch,path)
        input1_val = GeneratedDataset(epoch,path,'val')
        input2_val = InputDataset(epoch,path,'val')
        metrics_kwargs = {
                            'batch_size': 128,
                            'isc': True,
                            'fid': True,
                            'kid': True,
                            }
        metrics_dict[epoch]['train'] = calculate_metrics(input1 = input1_train,input2 = input2_train, **metrics_kwargs)
        metrics_dict[epoch]['val'] = calculate_metrics(input1 = input1_val,input2 = input2_val, **metrics_kwargs)
        print('##########################################################################################################################################################################')
    #save metrics_dict to json file
    metrics_path = os.path.join(path,'metrics.json')
    with open(metrics_path, 'w') as fp:
        json.dump(metrics_dict, fp)

if __name__ == '__main__':
    main()


