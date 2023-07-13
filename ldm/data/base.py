import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import cv2
from utils.mcmanager import MCManager

from abc import abstractmethod
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset



class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths,landmark_paths, aus, keys=None, size=225, mcManager=None):
        self.size = size
        self.keys = dict() if keys is None else keys
        self.keys["file_path_"] = paths
        self.keys["landmark_path_"] = landmark_paths
        self._length = len(paths)
        self.aus = list(aus)
        if mcManager:
            mcManager = MCManager()
        self._aligner = FaceAlign(size,mcManager=mcManager)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler,self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path,landmark_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        landmarks = np.load(landmark_path)
        image,landmarks = self._aligner(image,landmarks=landmarks,image_path=image_path)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image, landmarks

    def __getitem__(self, i):
        sample = dict()
        #sample["image"],sample['landmarks'] = self.preprocess_image(self.keys["file_path_"][i],self.keys["landmark_path_"][i])
        sample["image"],_ = self.preprocess_image(self.keys["file_path_"][i],self.keys["landmark_path_"][i])
        for k in self.keys:
            #sample[k] = self.keys[k][i]
            if k == "aus":
                sample[k] = self.keys[k][i]
                #get where the aus are 1
                inds = np.where(sample[k] == 1)[0]
                if len(inds) > 0:
                    sample["aus_humanlabel"] = (',').join([self.aus[ind] for ind in inds])
                else:
                    sample["aus_humanlabel"] = ''
        return sample
    

# code adapted from https://github.com/ZhiwenShao/PyTorch-JAANet/blob/master/dataset/face_transform.py
class FaceAlign():
    def __init__(self,img_size,enlarge = 2.9, mcManager=None):
        self.size = img_size
        self._enlarge = enlarge
        self.mcManager = mcManager

    def __call__(self, img, landmarks,image_path):
        key = image_path
        if self.mcManager is not None and key  in self.mcManager:
            result = self.mcManager.get(key)
            landmarks = np.array(result['landmarks'])
            mat = np.array(result['affine_mat'])
            aligned_img = cv2.warpAffine(img, mat[0:2, :], (self.size, self.size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))
        else:  
            if landmarks.shape[1] > 60:
                landmarks = landmarks[:,17:]
            left_eye_x  = landmarks[0,19:25].sum()/6.0
            left_eye_y  = landmarks[1,19:25].sum()/6.0
            right_eye_x = landmarks[0,25:31].sum()/6.0
            right_eye_y = landmarks[1,25:31].sum()/6.0
            dx = right_eye_x-left_eye_x
            dy = right_eye_y-left_eye_y
            l = (dx * dx + dy * dy)**0.5
            sinVal = dy / l
            cosVal = dx / l
            mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])
            mat2 = np.mat([[left_eye_x,left_eye_y,1],
                        [right_eye_x,right_eye_y,1],
                        [landmarks[0,13],landmarks[1,13],1],
                        [landmarks[0,31],landmarks[1,31],1],
                        [landmarks[0,37],landmarks[1,37],1]])
            mat2 = (mat1 * mat2.T).T
            cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
            cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5
            if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
                halfSize = 0.5 * self._enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
            else:
                halfSize = 0.5 * self._enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))
            scale = (self.size - 1) / 2.0 / halfSize
            mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
            mat = mat3 * mat1
            aligned_img = cv2.warpAffine(img, mat[0:2, :], (self.size, self.size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            land_3d = np.ones((landmarks.shape[0]+1,landmarks.shape[1]))
            land_3d[:-1,:] = landmarks
            landmarks = (mat*land_3d)[:-1]
            landmarks = np.asarray(landmarks).round()
            if self.mcManager is not None:
                try:
                    save_dict={'landmarks':landmarks,'affine_mat':mat}
                    self.mcManager.set(key,save_dict)
                except Exception:
                    pass
        return aligned_img,landmarks
    


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass
