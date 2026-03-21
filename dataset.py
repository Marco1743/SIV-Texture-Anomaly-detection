import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Importiamo le TUE funzioni
from augmentation import custom_histogram_equalization, apply_convolution, custom_gaussian_kernel

class MoCoTextureDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
        self.is_train = is_train

        # IL FIX: Aggiungiamo un Resize a 224x224 (Standard per ResNet)
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def apply_custom_augs(self, img_arr):
        aug_choice = random.choice([0, 1, 2])
        if aug_choice == 0:
            return custom_histogram_equalization(img_arr)
        elif aug_choice == 1:
            kernel = custom_gaussian_kernel(size=3, sigma=1.2)
            return apply_convolution(img_arr, kernel)
        else:
            return img_arr

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_pil = Image.open(img_path).convert('L')
        
        # Applichiamo il resize spaziale PRIMA di convertirla in array!
        img_pil = self.resize(img_pil)
        img_arr = np.array(img_pil)

        if self.is_train:
            view_1_arr = self.apply_custom_augs(img_arr)
            view_2_arr = self.apply_custom_augs(img_arr)
            
            view_1 = self.to_tensor(Image.fromarray(view_1_arr))
            view_2 = self.to_tensor(Image.fromarray(view_2_arr))
            
            return view_1, view_2
        else:
            return self.to_tensor(img_pil)