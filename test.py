import torch
from dataset.dataset import Dataset
from model.EdgeTraceNet import EdgeTraceNet
import torchvision.utils as vutils
import os
import random
from config import get_args

"""
    bring the best model and test it on the test dataset.
    save the result as a 3x3 grid image.
"""

args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = args.img_dir
gt_dir = args.gt_dir
model_save_path = args.model_save_path
img_save_path = args.img_save_path

test_dataset = Dataset(image_dir=img_dir, gt_dir=gt_dir, mode='test')

sample = random.choice(test_dataset)
norm_image = sample['norm_image'].to(device).unsqueeze(0)
raw_image = sample['raw_image'].to(device).unsqueeze(0)

model = EdgeTraceNet().to(device)

best_model_file = os.path.join(model_save_path, 'best_model_epoch.pth')
checkpoint = torch.load(best_model_file, map_location=device, weights_only=True)

model.load_state_dict(checkpoint['state_dict'])
model.eval()

outputs = model(norm_image)

edge = outputs['edges']

edge_right = 1.0 - edge[:, 0:1, :, :]
edge_down = 1.0 - edge[:, 1:2, :, :]

edge_map = torch.max(edge_right, edge_down) # [B, 1, H, W]
edge_map = (edge_map >= checkpoint['best_threshold']).float()

edge_map_rgb = edge_map.repeat(1, 3, 1, 1) # [B, 3, H, W]
combined = torch.cat([raw_image, edge_map_rgb], dim=3)

vutils.save_image(combined, os.path.join(img_save_path, f'vis_epoch_test.png'), nrow=2, padding=4)
print(f"saved V4 3x3 grid image: vis_epoch_test.png")