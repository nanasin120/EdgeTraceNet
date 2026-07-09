import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith('.jpg')
        ])
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(size=(224, 224))
        ])
        
        self.norm_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        raw_image = self.base_transform(Image.open(image_path).convert('RGB'))
        norm_image = self.norm_transform(raw_image)

        return {
            'raw_image' : raw_image,
            'norm_image' : norm_image
        }