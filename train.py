import torch
import os
import time
import torch.optim as optim
from eval import evaluate
from torch.utils.data import DataLoader
from dataset.dataset import Dataset
from model.EdgeTraceNet import EdgeTraceNet
from loss.loss import color_difference_loss, regularization_loss, feature_difference_loss, binarization_loss, thick_edge_loss
from utils.defs import save_image
from config import get_args

args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = args.batch_size
start_epoch = args.start_epoch
epochs = args.epochs
additional_epoch = epochs - start_epoch
lr = args.lr

weight_color = args.weight_color
weight_feature = args.weight_feature
weight_regular = args.weight_regular
weight_binary = args.weight_binary
weight_thick = args.weight_thick

image_save_interval = args.image_save_interval
weight_save_interval = args.weight_save_interval

img_dir = args.img_dir
gt_dir = args.gt_dir
model_save_path = args.model_save_path
img_save_path = args.img_save_path

if not os.path.exists(model_save_path): os.makedirs(model_save_path)
if not os.path.exists(img_save_path): os.makedirs(img_save_path)

# google colab
# img_dir = r"/content/data_local/BSDS500/images/"
# gt_dir = r"/content/data_local/BSDS500/ground_truth/"

train_dataset = Dataset(image_dir=img_dir, gt_dir=None, mode='train')
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
)

test_dataset = Dataset(image_dir=img_dir, gt_dir=gt_dir, mode='test')
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True
)

model = EdgeTraceNet().to(device)

criterion_color_diff = color_difference_loss().to(device)
criterion_regularization = regularization_loss().to(device)
criterion_feature_diff = feature_difference_loss().to(device)
criterion_binary = binarization_loss().to(device)
criterion_thick = thick_edge_loss().to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=additional_epoch, eta_min=1e-6)

def train():
    print('Train start')

    best_f1 = -1.0

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        train_loss = 0.0
        train_color_loss = 0.0
        train_feature_loss = 0.0
        train_regular_loss = 0.0
        train_binary_loss = 0.0
        train_thick_loss = 0.0

        epoch_start_time = time.time()
        batch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            RAW_IMAGE = batch['raw_image'].to(device)
            NORM_IMAGE = batch['norm_image'].to(device)

            OUTPUTS = model(NORM_IMAGE)
            
            FEATURES = OUTPUTS['features']
            EDGES = OUTPUTS['edges']

            loss_feature = criterion_feature_diff(EDGES, FEATURES.detach())

            loss_color = criterion_color_diff(EDGES, RAW_IMAGE)
            loss_regular = criterion_regularization(EDGES)
            loss_binary = criterion_binary(EDGES)
            loss_thick = criterion_thick(EDGES)

            total_loss = (loss_color * weight_color) + (loss_regular * weight_regular) + \
                (loss_feature * weight_feature) + (loss_binary * weight_binary) + (loss_thick * weight_thick)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss_total : {total_loss.item():.4f} Time : {batch_end_time-batch_start_time:.4f}")
                batch_start_time = time.time()

            train_loss += total_loss.item()
            train_color_loss += loss_color.item()
            train_regular_loss += loss_regular.item()
            train_feature_loss += loss_feature.item()
            train_binary_loss += loss_binary.item()
            train_thick_loss += loss_thick.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_color_loss = train_color_loss / len(train_loader)
        avg_train_regular_loss = train_regular_loss / len(train_loader)
        avg_train_feature_loss = train_feature_loss / len(train_loader)
        avg_train_binary_loss = train_binary_loss / len(train_loader)
        avg_train_thick_loss = train_thick_loss / len(train_loader)

        epoch_end_time = time.time()
        scheduler.step()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Color Loss : {avg_train_color_loss:.4f} Feature Loss : {avg_train_feature_loss:.4f} \n Regular Loss : {avg_train_regular_loss:.4f} Binary Loss : {avg_train_binary_loss:.4f} Thick Loss : {avg_train_thick_loss:.4f} Time : {epoch_end_time-epoch_start_time:.4f}')

        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()

                current_f1, current_th, current_prec, current_rec = evaluate(model, device, test_loader)

                print(f"👉 Epoch {epoch} 결과 - Best Threshold: {current_th:.1f} | Best F1-Score: {current_f1:.4f} | Best precision : {current_prec:.4f} | Best recall: {current_rec:.4f}")

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    
                    save_path = os.path.join(model_save_path, 'best_model_epoch.pth')
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'best_threshold': float(current_th),
                        'best_f1': float(best_f1),
                        'epoch': int(epoch)
                    }
                    torch.save(checkpoint, save_path)

                    print(f"New Best Model Saved with F1-Score: {best_f1:.4f} at Threshold {current_th:.1f}!")

        if epoch % args.weight_save_interval == 0:
            save_path = os.path.join(model_save_path, f'model_epoch_{epoch}.pth')
            
            checkpoint = {
                'state_dict': model.state_dict(),
                'best_threshold': float(current_th),
                'best_f1': float(best_f1),
                'epoch': int(epoch)
            }
            torch.save(checkpoint, save_path)

            print(f'Saved : {model_save_path}')

        if epoch % args.image_save_interval == 0:
            save_image(epoch, EDGES, RAW_IMAGE, img_save_path, current_th)


if __name__ == "__main__":
    train()