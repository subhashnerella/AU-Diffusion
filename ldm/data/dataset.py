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


class BP4D(FacesBase):
    def __init__(self,aus,split=None,size=225,mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('ldm/data/datafiles/bp4d.csv'))
        if split is not None:
            df = helper_split_func(df,split=split)
        relpaths = df['path'].values
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels, 'dataset':'BP4D' }
        self.data = ImagePaths(paths,aus,landmark_paths,labels,size,mcManager)

class BP4DPlus(FacesBase):
    def __init__(self,aus,split=None,size=225,mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('ldm/data/datafiles/bp4d+.csv'))
        if split is not None:
            df = helper_split_func(df,split=split)
        relpaths = df['path'].values
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels, 'dataset':'BP4DPlus' }
        self.data = ImagePaths(paths,aus,landmark_paths,labels,size,mcManager)

class DISFA(FacesBase):
    def __init__(self,aus,split=None,size=225,mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('ldm/data/datafiles/disfa.csv'))
        if split is not None:
            df = helper_split_func(df,split=split)
        relpaths = df['path'].values
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels, 'dataset':'DISFA' }
        self.data = ImagePaths(paths,aus,landmark_paths,labels,size,mcManager)


class UNBC(FacesBase):
    def __init__(self,aus,split=None,size=225,mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('ldm/data/datafiles/unbc.csv'))
        if split is not None:
            df = helper_split_func(df,split=split)
        relpaths = df['path'].values
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels, 'dataset':'UNBC' }
        self.data = ImagePaths(paths,aus,landmark_paths,labels,size,mcManager)


class MultiDataset(Dataset):
    def __init__(self, datasets,aus,split=None,size=225,mcManager=None):
        dataset_classes = {'BP4D': BP4D,
                           'DISFA': DISFA,
                           'UNBC': UNBC,
                           'BP4DPlus': BP4DPlus}
        dataset = []
        for d in datasets:
            dataset.append(dataset_classes[d](aus,split,size=size,mcManager=mcManager))
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
    au_df = au_df[aus].astype(int)
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
        #df = df.sample(frac=1,random_state=42)
        print(np.sort(df.participant.unique().tolist()))
        print(len(df))
    return df