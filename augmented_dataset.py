import os, sys
from PIL import Image
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset

"""
    This class allows to generate the image and its corresponding groundtruth.
    If we put is_transform = True, we augment the dataset on the fly with random 
    transformation like flip, rotation and crop in random the images to get a size 
    of 303x304 pixels.
    
    @param file_path_data: the path to the directory containing the images
    @param file_path_label: the path to the directory containing the grundtruths
    @param is_transform: if True, augment the dataset
"""
class AugmentedDataset(Dataset):
    def __init__(self, file_path_data, file_path_label, is_transform=True):
        self.data_names = os.listdir(file_path_data)
        self.path_data = file_path_data
        
        self.label_names = os.listdir(file_path_label)
        self.path_label = file_path_label
        
        self.is_transform = is_transform
        
    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        file_path_data = os.path.join(self.path_data, self.data_names[idx])
        file_path_label = os.path.join(self.path_label, self.label_names[idx])
        
        image = Image.open(file_path_data)
        label = Image.open(file_path_label)

        if self.is_transform:
            image, label = self.transform(image, label)

        return image, label
    
    def transform(self, image, mask):
        
        # Random crop (208x208)
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(304, 304))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        angle = random.choice([-90, -45, 0, 45, 90])
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        return image, mask