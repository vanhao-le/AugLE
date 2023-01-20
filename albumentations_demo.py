'''
Albumentations requires Python 3.6 or higher.
Install the latest stable version from PyPI
pip install -U albumentations
'''
from email.mime import image
import os
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data as data
from torch.utils.data import  DataLoader
import torchvision.models as models
from torchvision import transforms
import pandas as pd
from torchvision.utils import save_image
import cv2
import numpy as np

cudnn.benchmark = True

class AlbumDS(data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.file_list = list()
        for f in os.listdir(self.image_paths):
            file_name, file_ext = os.path.splitext(f)            
            if file_ext == '.jpg':
                self.file_list.append(f)
        

    def __getitem__(self, index):
        input_image_path = os.path.join(self.image_paths, self.file_list[index])

        image = Image.open(input_image_path)

        if self.transform:
            # Convert PIL image to numpy array
            image_np = np.array(image)
            # Apply transformations
            augmented = self.transform(image=image_np)
            # Convert numpy array to PIL Image
            image = Image.fromarray(augmented['image'])            
            aug_transform = transforms.Compose(base_transforms)
            image = aug_transform(image)

        else:
            aug_transform = transforms.Compose(base_transforms)
            image = aug_transform(image)
        
                
        return image, self.file_list[index]

    def __len__(self):
        return len(self.file_list)

# The augmentations we want to test robustness to

augmentations = {
    "301": A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    "302": A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    "303": A.NoOp(p=1),
    "304": A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15, p=1),
    "305": A.HueSaturationValue(hue_shift_limit=5,sat_shift_limit=5, p=1),
    "306": A.HorizontalFlip(p=1.0),
    "307": A.VerticalFlip(p=1.0),
    "308": A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    "309": A.ToGray(p=1.0),
    "310": A.RandomGridShuffle(p=1.0),
    "311": A.NoOp(p=1),
    "312": A.NoOp(p=1),
    "313": A.NoOp(p=1),
    "314": A.NoOp(p=1),
    "315": A.NoOp(p=1),
    "316": A.NoOp(p=1),
    "317": A.ChannelDropout(p=1),
    "318": A.CoarseDropout(max_holes=3, max_width=100, max_height=100, min_height=16, min_width=16, p=1),
    "319": A.RandomBrightnessContrast(p=1.0),
    "320": A.NoOp(p=1),
    "321": A.NoOp(p=1),
    "322": A.NoOp(p=1),
    "323": A.NoOp(p=1),
    "324": A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=1.0),
    "325": A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=1.0),
    "326": A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0),
    "327": A.RandomToneCurve(scale=0.5,p=1),
    "328": A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2, p=1),
    "329": A.MotionBlur(blur_limit=33, p=1),
    "330": A.GaussNoise(var_limit=(0, 255), p=1),
    "331": A.MedianBlur(blur_limit=3, p=1),
    "332": A.Blur(blur_limit=7, p=1),
    "333":  A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, alpha_coef=0.1, p=1),
    "334": A.Perspective(p=1),   
    "335": A.NoOp(p=1),
    "336": A.NoOp(p=1),
    "337": A.NoOp(p=1),
    "338": A.NoOp(p=1),
    "339": A.NoOp(p=1),
    "340": A.NoOp(p=1),    
}

# Processing transformations which will be applied to all images
base_transforms = [    
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
]

BATCH_SIZE = 256

def data_augmentation():

    reference_csv = r'data\category.csv'
    db_root = r'D:\VSC\train\reference'
    aug_path = r'D:\VSC\train\aug'
    image_path = r'D:\VSC\train\frame'
    

    df = pd.read_csv(reference_csv)
    
    data = []    
    for item in df.itertuples():
        video_id = item.class_name
        class_id = item.classIDx

        count = 301
        for aug_name, aug in augmentations.items():      
        
            new_id = video_id + '-' + str(count)
            ouput_path = os.path.join(aug_path, new_id)
            if not os.path.exists(ouput_path):
                os.makedirs(ouput_path)            

            input_path = os.path.join(image_path, video_id)

            input_dataset =  AlbumDS(image_paths=input_path, transform=aug)            
            input_loader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)            
            
            for images, names in input_loader:
                for i in range(0, len(names)):
                    output_image_path = os.path.join(ouput_path, str(names[i]))                    
                    save_image(images[i], output_image_path)
                                
            case = {'video_id': new_id, 'classIDx': class_id}
            data.append(case)
            count += 1

              

    df = pd.DataFrame(data)
    df.to_csv(r'data\album-100.csv', index=False)

    print()

def duplicate_data(start_num, end_num):

    reference_csv = r'data\category.csv'
    db_root = r'D:\VSC\train\reference'
    aug_path = r'D:\VSC\train\aug'
    image_path = r'D:\VSC\train\frame'
    

    df = pd.read_csv(reference_csv)
    
    data = []
    
    for item in df.itertuples():
        video_id = item.class_name
        class_id = item.classIDx

        count = start_num
        for i in range(start_num, end_num):      
        
            new_id = video_id + '-' + str(count)
            ouput_path = os.path.join(aug_path, new_id)
            if not os.path.exists(ouput_path):
                os.makedirs(ouput_path)            

            input_path = os.path.join(image_path, video_id)

            input_dataset =  AlbumDS(image_paths=input_path, transform=None)            
            input_loader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)            
            
            for images, names in input_loader:
                for i in range(0, len(names)):
                    output_image_path = os.path.join(ouput_path, str(names[i]))                    
                    save_image(images[i], output_image_path)
                                
            case = {'video_id': new_id, 'classIDx': class_id}
            data.append(case)
            count += 1

              

    df = pd.DataFrame(data)
    df.to_csv(r'data\duplicate-100.csv', index=False)

    return

def main():

    # duplicate_data(start_num=160, end_num=301)
    data_augmentation()
    # duplicate_data(start_num=341, end_num=450)

    return

if __name__ == '__main__':
    main()
    