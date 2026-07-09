import torch
import os
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import Dataset
from model.EdgeTraceNet import EdgeTraceNet
from loss.loss import color_difference_loss, regularization_loss, feature_difference_loss, binarization_loss, thick_edge_loss
from utils.defs import save_image

model_save_path = r'./save/model_save'
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
img_save_path = r'./save/image_save'
if not os.path.exists(img_save_path): os.makedirs(img_save_path)

BATCH = 16
START_EPOCH = 0
END_EPOCH = 500
ADDITIONAL_EPOCH = END_EPOCH-START_EPOCH
LEARNING_RATE = 1e-4
IMAGE_SAVE_INTERVEL = 10
WEIGHT_SAVE_INTERVEL = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r"data/BSDS500/images/train"
# img_dir = r"/content/data_local/BSDS500/images/train/"

full_dataset = Dataset(image_dir=img_dir)

dataloader = DataLoader(
    dataset=full_dataset,
    batch_size=BATCH,
    shuffle=True,
    pin_memory=True
)

model = EdgeTraceNet().to(DEVICE)

criterion_color_diff = color_difference_loss().to(DEVICE)
criterion_regularization = regularization_loss().to(DEVICE)
criterion_feature_diff = feature_difference_loss().to(DEVICE)
criterion_binary = binarization_loss().to(DEVICE)
criterion_thick = thick_edge_loss().to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADDITIONAL_EPOCH, eta_min=1e-6)

def train():
    print('Train start')
    best_avg_loss = float('inf')
    
    weight_color = 1.0
    weight_feature = 1.0
    weight_regular = 0.001
    weight_binary = 0.001
    weight_thick = 0.1

    for epoch in range(START_EPOCH, END_EPOCH + 1):
        model.train()

        train_loss = 0.0
        train_color_loss = 0.0
        train_feature_loss = 0.0
        train_regular_loss = 0.0
        train_binary_loss = 0.0
        train_thick_loss = 0.0

        epoch_start_time = time.time()
        batch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            RAW_IMAGE = batch['raw_image'].to(DEVICE)
            NORM_IMAGE = batch['norm_image'].to(DEVICE)

            OUTPUTS = model(NORM_IMAGE)
            
            FEATURES = OUTPUTS['features']
            EDGES = OUTPUTS['edges']

            loss_feature = criterion_feature_diff(EDGES[-1], FEATURES[-1].detach())
            
            # loss_feature = 0.0
            # for edge, feature in zip(EDGES, FEATURES):
            #     loss_feature += criterion_feature_diff(edge, feature.detach())
            # loss_feature = loss_feature / 4

            loss_color = criterion_color_diff(EDGES[-1], RAW_IMAGE)
            loss_regular = criterion_regularization(EDGES[-1])
            loss_binary = criterion_binary(EDGES[-1])
            loss_thick = criterion_thick(EDGES[-1])

            total_loss = (loss_color * weight_color) + (loss_regular * weight_regular) + \
                (loss_feature * weight_feature) + (loss_binary * weight_binary) + (loss_thick * weight_thick)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f"Epoch [{epoch}/{END_EPOCH}] Batch [{batch_idx}/{len(dataloader)}] Loss_total : {total_loss.item():.4f} Time : {batch_end_time-batch_start_time:.4f}")
                batch_start_time = time.time()

            train_loss += total_loss.item()
            train_color_loss += loss_color.item()
            train_regular_loss += loss_regular.item()
            train_feature_loss += loss_feature.item()
            train_binary_loss += loss_binary.item()
            train_thick_loss += loss_thick.item()

        avg_train_loss = train_loss / len(dataloader)
        avg_train_color_loss = train_color_loss / len(dataloader)
        avg_train_regular_loss = train_regular_loss / len(dataloader)
        avg_train_feature_loss = train_feature_loss / len(dataloader)
        avg_train_binary_loss = train_binary_loss / len(dataloader)
        avg_train_thick_loss = train_thick_loss / len(dataloader)

        epoch_end_time = time.time()
        scheduler.step()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Color Loss : {avg_train_color_loss:.4f} Feature Loss : {avg_train_feature_loss:.4f} \n Regular Loss : {avg_train_regular_loss:.4f} Binary Loss : {avg_train_binary_loss:.4f} Thick Loss : {avg_train_thick_loss:.4f} Time : {epoch_end_time-epoch_start_time:.4f}')

        if epoch % WEIGHT_SAVE_INTERVEL == 0:
            save_path = os.path.join(model_save_path, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

            print(f'Saved : {model_save_path}')

        if epoch % IMAGE_SAVE_INTERVEL == 0:
            save_image(epoch, EDGES[-1], RAW_IMAGE, img_save_path)

        if avg_train_loss < best_avg_loss:
            best_avg_loss = avg_train_loss
            save_path = os.path.join(model_save_path, f'best_model_epoch.pth')
            torch.save(model.state_dict(), save_path)

            print(f'New Best Model Saved! Loss : {best_avg_loss:.4f}') 

if __name__ == "__main__":
    train()