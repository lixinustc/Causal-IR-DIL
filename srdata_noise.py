import os
import glob

import numpy as np
import torch
import torch.utils.data as data
import random
import cv2


class DataCrop(data.Dataset):
    def __init__(self, choose, hr_folder, patch_size=64):
        self.patch_size = patch_size
        self.dir_hr = hr_folder
        self.images_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*.png')))
        self.choose = (choose + 1) * 5  # 5, 10, 15, 20

    def __getitem__(self, idx):
        filename = self.images_hr[idx].split('/')[-1]

        hr = cv2.imread(os.path.join(self.dir_hr, filename))  # BGR, n_channels=3        
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)  # RGB, n_channels=3

        croph = np.random.randint(0, 256 - self.patch_size)
        cropw = np.random.randint(0, 256 - self.patch_size)
        hr = hr[croph: croph+self.patch_size, cropw: cropw+self.patch_size, :]

        mode = np.random.randint(0, 8)
        hr = augment_img(hr, mode=mode)

        hr = hr.astype(np.float32) / 255.
        lr = hr.copy()

        noise = np.random.randn(*hr.shape) * self.choose / 255.

        lr += noise

        lr = np.clip(lr, 0, 1).astype(np.float32)

        lr = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1)))
        hr = torch.from_numpy(np.ascontiguousarray(hr.transpose(2, 0, 1)))

        return lr, hr

    def __len__(self):
        return len(self.images_hr)


class DataTest(data.Dataset):
    def __init__(self, hr_folder='default', level=50):

        self.dir_hr = 'Set5/HR' if hr_folder == 'default' else hr_folder
        self.name_hr = sorted(os.listdir(self.dir_hr))
        self.level = level

    def __getitem__(self, idx):
        name = self.name_hr[idx]
        hr = cv2.cvtColor(cv2.imread(os.path.join(self.dir_hr, name)), cv2.COLOR_BGR2RGB)

        hr = hr.astype(np.float32) / 255.
        lr = hr.copy()

        noise = np.random.randn(*hr.shape) * self.level / 255.

        lr += noise

        lr = np.clip(lr, 0, 1).astype(np.float32)

        lr = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1)))
        hr = torch.from_numpy(np.ascontiguousarray(hr.transpose(2, 0, 1)))

        return lr, hr, name

    def __len__(self):
        return len(self.name_hr)


def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))