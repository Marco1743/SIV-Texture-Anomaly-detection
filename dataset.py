import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Importiamo le TUE funzioni (ricorda: file al singolare!)
from augmentation import custom_histogram_equalization, apply_convolution, custom_gaussian_kernel

class MoCoTextureDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        """
        data_dir: Il percorso della cartella (es. 'data/train/good')
        is_train: Se True, genera le due viste per MoCo. Se False, restituisce l'immagine singola per il test.
        """
        self.data_dir = data_dir
        # Prendiamo tutti i file .png nella cartella
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
        self.is_train = is_train

        # PyTorch lavora con i Tensori. Usiamo transforms solo per la conversione matematica finale (da NumPy a Tensor [0, 1])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def apply_custom_augs(self, img_arr):
        """
        Sceglie casualmente una delle tue Data Augmentation scritte 'from scratch'
        per alterare leggermente la texture.
        """
        # Scegliamo a caso: 0 = Equalizzazione, 1 = Blur, 2 = Nessuna modifica
        aug_choice = random.choice([0, 1, 2])
        
        if aug_choice == 0:
            return custom_histogram_equalization(img_arr)
        elif aug_choice == 1:
            kernel = custom_gaussian_kernel(size=3, sigma=1.2)
            return apply_convolution(img_arr, kernel)
        else:
            return img_arr

    def __getitem__(self, idx):
        # 1. Carichiamo l'immagine base
        img_path = self.image_paths[idx]
        img_pil = Image.open(img_path).convert('L')
        img_arr = np.array(img_pil)

        if self.is_train:
            # MAGIA MOCO: Generiamo DUE viste dinamicamente per il Contrastive Learning
            view_1_arr = self.apply_custom_augs(img_arr)
            view_2_arr = self.apply_custom_augs(img_arr)
            
            # Convertiamo da NumPy a PIL e poi a Tensor PyTorch
            view_1 = self.to_tensor(Image.fromarray(view_1_arr))
            view_2 = self.to_tensor(Image.fromarray(view_2_arr))
            
            return view_1, view_2
        else:
            # In fase di test (quando cercheremo le anomalie), ci serve solo l'immagine nuda e cruda
            return self.to_tensor(img_pil)