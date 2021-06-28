from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch


class MatNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.palette = {
            (0,0,0):0,
            (114,114,223):7,
            (250,50,183):1,
            (255,204,51):2,
            (36,179,83):0,
            (118,211,58):0,
            (184,61,245):2,
            (52,209,183):3,
            (212,103,122):2,
            (129,129,129):4,
            (120,36,121):0,
            (185,123,7):5,
            (255,0,0):6,
            (138,6,7):0,
            (71,51,163):7,
            (42,125,209):0,
            (245,147,49):8
        }
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        
        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)
        
        image = np.resize(image,(512, 512, 3))
        mask = np.resize(mask,(512, 512, 3))
        
        
        output = np.zeros([mask.shape[0], mask.shape[1],9])
        for k in self.palette.keys():
                output[:,:,self.palette[k]] = (mask==k)[:,:,0].astype(np.uint8)
        


        if self.transform is not None:
            augmentations = self.transform(image=image, mask=output)
            image = augmentations["image"]
            output = augmentations["mask"]
            
            
        mask = output.transpose(2,0,1).astype(np.float32)
        image = image.transpose(2,0,1).astype(np.float32)
        
        
        return image, mask


