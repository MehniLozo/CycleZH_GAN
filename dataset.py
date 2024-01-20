from PIL import Image 
import os
from torch.utils.data import Dataset 
import numpy as np

class HZDataset(Dataset):
    def __init__(self,root_z,root_h,transform = None):
        self.root_z = root_z 
        self.root_h = root_h
        self.transform = transform
        self.z_ims = os.listdir(root_z)
        self.h_ims = os.listdir(root_h)
        self.z_len = len(self.z_ims)
        self.h_len = len(self.h_img)
        self.len_data = max(self.z_len,self.h_len)
    
    def __len__(self):
        return self.len_data
    def __getitem__(self,index):
        z_im = self.z_ims[index%self.z_len]
        h_im = self.h_ims[index%self.h_len]

        z_path = os.path.join(self.root_z,z_im)
        h_path = os.path.join(self.root_h,h_im)

        z_im = np.array(Image.open(z_path).convert("RGB"))
        h_im = np.array(Image.open(h_path).convert("RGB"))

        if self.transform:
            aug = self.transform(image=z_im,image0 = h_im)
            z_im = aug["image"]
            h_im = aug["image0"]
        
        return z_im, h_im