import os
from torch.utils.data import Dataset
import pandas as pd
from ldm.data.base import ImagePaths, ConcatDatasetWithIndex
import numpy as np

ROOT = "/blue/parisa.rashidi/subhashnerella/Datasets/"


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        self.data = None

    def _load(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

class BP4DTrain(FacesBase):
    def __init__(self,aus,size=225,mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('ldm/data/datafiles/bp4d.csv'))
        df = helper_split_func(df)
        relpaths = df['path'].values
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels }
        self.data = ImagePaths(paths,landmark_paths,aus,labels,size,mcManager)

class BP4DVal(FacesBase):
    def __init__(self,aus,size=225,mcManager=None):
        super().__init__(size)
        df = pd.read_csv(os.path.join('ldm/data/datafiles/bp4d.csv'))
        df = helper_split_func(df,split='val')
        relpaths = df['path'].values
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels }
        self.data = ImagePaths(paths,landmark_paths,aus,labels,size,mcManager)

class DISFATrain(FacesBase):
    def __init__(self,aus,size=225,mcManager=None):
        super().__init__(size)
        df = pd.read_csv(os.path.join('ldm/data/datafiles/disfa.csv'))
        df = helper_split_func(df)
        relpaths = df['path'].values
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels }
        self.data = ImagePaths(paths,landmark_paths,aus,labels,size,mcManager)


class DISFAVal(FacesBase):
    def __init__(self,aus,size=225,mcManager=None):
        super().__init__(size)
        df = pd.read_csv(os.path.join('ldm/data/datafiles/disfa.csv'))
        df = helper_split_func(df,split='val')
        relpaths = df['path'].values
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels }
        self.data = ImagePaths(paths,landmark_paths,aus,labels,size,mcManager)


class MultiDatasetTrain(Dataset):
    def __init__(self, datasets,aus, size=225,mcManager=None):
        dataset_classes = {'BP4D': BP4DTrain,
                           'DISFA': DISFATrain}
        dataset = []
        for d in datasets:
            dataset.append(dataset_classes[d](aus,size=size,mcManager=mcManager))
        self.dataset = ConcatDatasetWithIndex(dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample,dataset_label = self.dataset[idx]
        sample['dataset_label'] = dataset_label
        return sample
    

class MultiDatasetVal(Dataset):
    def __init__(self, datasets,aus, size=225,mcManager=None):
        dataset_classes = {'BP4D': BP4DVal,
                           'DISFA': DISFAVal}
        dataset = []
        for d in datasets:
            dataset.append(dataset_classes[d](aus,size=size,mcManager=mcManager))
        self.dataset = ConcatDatasetWithIndex(dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample,dataset_label = self.dataset[idx]
        sample['dataset_label'] = dataset_label
        return sample


def helper_AU_func(df:pd.DataFrame,aus:list)->pd.DataFrame:
    au_df = df.filter(regex='AU*',axis=1)
    present_aus = au_df.columns.to_list()
    to_remove = list(set(present_aus) - set(aus))
    au_df = au_df.drop(columns=to_remove)
    absent_aus = list(set(aus) - set(present_aus))
    # Add absent AUs fillled with -1
    for au in absent_aus:
        au_df[au] = -1
    return au_df
        
def helper_split_func(df:pd.DataFrame,split:str = 'train')->pd.DataFrame:
    np.random.seed(42)
    participants = df['participant'].unique()
    participants = np.random.choice(participants,size=int(len(participants)*0.75),replace=False)
    
    if split == 'train':
        df = df[df['participant'].isin(participants)]
        print(np.sort(df.participant.unique().tolist()))
    else:
        df = df[~df['participant'].isin(participants)]
        print(np.sort(df.participant.unique().tolist()))
    return df