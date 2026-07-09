import torch
from dataset.dataset import Dataset
from model.EdgeTraceNet import EdgeTraceNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r"data/BSDS500/images/train"
full_dataset = Dataset(image_dir=img_dir)

sample = full_dataset[0]
image = sample['norm_image'].to(DEVICE)

model = EdgeTraceNet().to(DEVICE)

outputs = model(image.unsqueeze(0))

for output in outputs['edges']:
    print(output.shape)