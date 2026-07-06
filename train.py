import torch
import os
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import Dataset
from model.EdgeTraceNet import EdgeTraceNet
from loss.loss import color_difference_loss, regularization_loss, thick_edge_loss, smooth_loss
from utils.defs import save_image

model_save_path = r'./save/model_save'
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
img_save_path = r'./save/image_save'
if not os.path.exists(img_save_path): os.makedirs(img_save_path)

BATCH = 4
START_EPOCH = 0
END_EPOCH = 30
ADDITIONAL_EPOCH = END_EPOCH-START_EPOCH
LEARNING_RATE = 1e-4
IMAGE_SAVE_INTERVEL = 5
WEIGHT_SAVE_INTERVEL = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r'./data'

full_dataset = Dataset(image_dir=img_dir)

dataloader = DataLoader(
    dataset=full_dataset,
    batch_size=BATCH,
    shuffle=True,
    pin_memory=True
)

model = EdgeTraceNet().to(DEVICE)

criterion_color_diff = color_difference_loss().to(DEVICE)
criterion_refularization = regularization_loss().to(DEVICE)
criterion_thick_edge = thick_edge_loss().to(DEVICE)
criterion_smooth = smooth_loss().to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADDITIONAL_EPOCH, eta_min=1e-6)

def train():
    print('Train start')
    best_avg_loss = float('inf')

    for epoch in range(START_EPOCH, END_EPOCH + 1):
        model.train()

        train_loss = 0.0
        train_color_loss = 0.0
        train_regular_loss = 0.0
        train_thick_loss = 0.0
        train_smooth_loss = 0.0

        weight_color = 1.0
        weight_regular = 0.0
        weight_thick = 0.0
        weight_smooth = 0.0

        epoch_start_time = time.time()
        batch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            IMAGE = batch['image'].to(DEVICE)

            OUTPUTS = model(IMAGE)

            loss_color = criterion_color_diff(OUTPUTS, IMAGE)
            loss_regular = criterion_refularization(OUTPUTS)
            loss_thick = criterion_thick_edge(OUTPUTS)
            loss_smooth = criterion_smooth(OUTPUTS)

            total_loss = (loss_color * weight_color) + (loss_regular * weight_regular) \
                + (loss_thick * weight_thick) + (loss_smooth * weight_smooth)

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
            train_thick_loss += loss_thick.item()
            train_smooth_loss += loss_smooth.item()

        avg_train_loss = train_loss / len(dataloader)
        avg_train_color_loss = train_color_loss / len(dataloader)
        avg_train_regular_loss = train_regular_loss / len(dataloader)
        avg_train_thick_loss = train_thick_loss / len(dataloader)
        avg_train_smooth_loss = train_smooth_loss / len(dataloader)

        epoch_end_time = time.time()
        scheduler.step()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Color Loss : {avg_train_color_loss:.4f} Regular Loss : {avg_train_regular_loss:.4f} Thick Loss : {avg_train_thick_loss:.4f} Smooth Loss : {avg_train_smooth_loss:.4f} Time : {epoch_end_time-epoch_start_time:.4f}')

        if epoch % WEIGHT_SAVE_INTERVEL == 0:
            save_path = os.path.join(model_save_path, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

            print(f'Saved : {model_save_path}')

        if epoch % IMAGE_SAVE_INTERVEL == 0:
            save_image(epoch, OUTPUTS, IMAGE, img_save_path)

        if avg_train_loss < best_avg_loss:
            best_avg_loss = avg_train_loss
            save_path = os.path.join(model_save_path, f'best_model_epoch.pth')
            torch.save(model.state_dict(), save_path)

            print(f'New Best Model Saved! Loss : {best_avg_loss:.4f}') 

if __name__ == "__main__":
    train()