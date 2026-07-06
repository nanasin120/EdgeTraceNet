import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.Resize((704, 1248)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = self.transform(Image.open(image_path).convert('RGB'))

        return {
            'image' : image
        }