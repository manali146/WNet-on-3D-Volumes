# Setup Data Loading

import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import zoom

class PhantomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir) # List of all the images

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = nib.load(img_path).get_fdata()#.convert("RGB")

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        
        return image

# Removing unneccessary slices:
def remove_slices_3d_volume(volume, start_slice, end_slice):
    if start_slice < 0:
        start_slice = 0
    if end_slice >= volume.shape[0]:
        end_slice = volume.shape[0] - 1

    #sliced_volume = volume[start_slice:end_slice + 1]
    sliced_volume = volume[:, start_slice:251, :, :]
    #print('sliced_volume', sliced_volume.shape)
    return sliced_volume

# Function to adjust contrast
def adjust_contrast(image, contrast_range=(-200, 300)):
    # Contrast range specifies the window in Hounsfield units (HU) to display.
    image = np.clip(image, contrast_range[0], contrast_range[1])
    # Scale the image to the range [0, 1]
    image = (image - contrast_range[0]) / (contrast_range[1] - contrast_range[0])
    return image

# Function to normalize image to the range [0, 1]
def normalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if image.size == 0:
        return image
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def preprocessing_transform(cr_img_data, device):
    # Transpose the data
    print('image shape ------------->',cr_img_data.shape)
    t_data = cr_img_data#np.transpose(cr_img_data, (0, 2, 3, 1))
    #print(t_data.shape)

    # Remove unnecessary slices
    start_slice = 41
    end_slice = 251
    sliced_volume = remove_slices_3d_volume(t_data, start_slice, end_slice)
    print('sliced_volume ------------->',sliced_volume.shape)
    # Adjust contrast
    contrasted_image = adjust_contrast(sliced_volume, contrast_range=(-300, 3000))

    # Normalize image
    normalized_image = normalize_image(contrasted_image)
    #print('normalized_image---------',normalized_image.shape)
    # Zoom the image
    scan = zoom(normalized_image, (80 / 130))

    # Preprocess the data
    data = scan.astype(np.float16)
    print('zoomed data ------------->',data.shape)
    # Remove unnecessary dimensions
    data = np.squeeze(data)
    #print(data.shape)
    data = np.expand_dims(data, axis=0)
    #print(data.shape)
    data = np.expand_dims(data, axis=1)
    #print(data.shape)
    # Create a PyTorch tensor
    processed_data = torch.tensor(data, device=device)
    #print('processed_data -------------> ',processed_data.shape)
    return processed_data
