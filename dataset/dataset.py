import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Dataset(Dataset):
    """
        it has two modes: train and test. 
        In train mode, it returns the raw image, normalized image, and the path to the ground truth .mat file. 
        In test mode, it returns the raw image and normalized image only. At this tiime, it returns the image in the original size.
    """
    def __init__(self, image_dir, gt_dir, mode='train'):
        self.mode = mode.lower()
        self.image_dir = os.path.join(image_dir, self.mode)
        self.gt_dir = os.path.join(gt_dir, self.mode) if gt_dir else None

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir) 
            if f.lower().endswith('.jpg')
        ])
        
        if self.mode == 'train':
            self.base_transform = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor()
            ])
        else:
            self.base_transform = transforms.ToTensor()
            
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

        data = {
            'raw_image': raw_image,
            'norm_image': norm_image,
            'filename': image_name,
        }

        if self.gt_dir:
            gt_name = os.path.splitext(image_name)[0] + '.mat'
            gt_path = os.path.join(self.gt_dir, gt_name)
            data['gt_path'] = gt_path

        return data