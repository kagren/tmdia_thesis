import random
import torch
from PIL import Image
import numpy as np
import cv2
import albumentations as A
import albumentations.pytorch.transforms as APT

class BreastCancerDataSet_Basic(torch.utils.data.Dataset):
    
    def __init__(self, df, path, TARGET, transforms=None, use_patient_id_for_path = True):
        super().__init__()
        
        self.df = df
        
        self.path = path
        self.transforms = transforms
        self.TARGET = TARGET
        self.use_patient_id_for_path = use_patient_id_for_path

    def load_image(self, i):
        
        if self.use_patient_id_for_path:
            pth = f'{self.path}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
        else:
            pth = f'{self.path}/{self.df.iloc[i].image_id}.png'

        try:
            img = Image.open(pth) #.convert('RGB')
            
        except Exception as ex:
            print(pth, ex)
            return None
        
        return img
            
    def get_labels(self, i):

        label_cancer = torch.as_tensor(self.df.iloc[i][self.TARGET]).float()
        
        return label_cancer

    def __getitem__(self, i):

        img = self.load_image(i)

        if self.transforms is not None:
            img = self.transforms(img)
            
        if self.TARGET not in self.df.columns:
            return img

        return img, self.get_labels(i)

        
    def __len__(self):
        return len(self.df)

class BreastCancerDataSet_Mixup(torch.utils.data.Dataset):
    
    def __init__(self, df, path, TARGET, transforms=None):
        super().__init__()
        
        self.df = df
        
        self.path = path
        self.transforms = transforms
        self.TARGET = TARGET

    def load_image(self, i):
        
        pth = f'{self.path}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
        
        try:
            img = Image.open(pth) #.convert('RGB')
            
        except Exception as ex:
            print(pth, ex)
            return None
        
        return img
            
    def get_labels(self, i):

        label_cancer = torch.as_tensor(self.df.iloc[i][self.TARGET]).float()
        
        return label_cancer

    def __getitem__(self, i):

        i_mixup = np.random.randint(len(self.df))
        
        img = self.load_image(i)
        img_mixup = self.load_image(i_mixup)

        if self.transforms is not None:
            img = self.transforms(img)
            img_mixup = self.transforms(img_mixup)
            
        if self.TARGET not in self.df.columns:
            return img, img_mixup

        return img, img_mixup, self.get_labels(i)

        
    def __len__(self):
        return len(self.df)


class BreastCancerDataSet_MultiLabel(torch.utils.data.Dataset):
    
    def __init__(self, df, path, transforms=None, use_patient_id_for_path = True):
        super().__init__()
        
        self.df = df
        
        self.path = path
        self.transforms = transforms
        self.use_patient_id_for_path = use_patient_id_for_path

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        self.one_hot_subtype = torch.as_tensor(le.fit_transform(df.subtype))
        self.one_hot_abnormality = torch.as_tensor(le.fit_transform(df.abnormality))

    def load_image(self, i):
        
        if self.use_patient_id_for_path:
            pth = f'{self.path}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
        else:
            pth = f'{self.path}/{self.df.iloc[i].image_id}.png'

        try:
            img = Image.open(pth) #.convert('RGB')
            
        except Exception as ex:
            print(pth, ex)
            return None
        
        return img
            
    def get_labels(self, i):

        label_cancer = torch.as_tensor(self.df.iloc[i].y).float()
        label_subtype = self.one_hot_subtype[i]
        label_abnormality = self.one_hot_abnormality[i]
        
        return label_cancer, label_subtype, label_abnormality

    def __getitem__(self, i):

        img = self.load_image(i)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.get_labels(i)

        
    def __len__(self):
        return len(self.df)