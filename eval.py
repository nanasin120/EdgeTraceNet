import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils.defs import load_gt_boundaries
from config import get_args
from dataset.dataset import Dataset
from model.EdgeTraceNet import EdgeTraceNet

def evaluate(model=None, device=None, test_loader=None):
    args = get_args()
    img_dir = args.img_dir
    gt_dir = args.gt_dir

    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        print(f"⚠️ test path are invalid. check the paths \nImg: {img_dir}\nGT: {gt_dir}")
        return

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        model = EdgeTraceNet().to(device)
        best_model_file = os.path.join(args.model_save_path, 'best_model_epoch.pth')
        
        if os.path.exists(best_model_file):
            checkpoint = torch.load(best_model_file, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])

            print(f"✅ Successfully loaded weights: {best_model_file}")
        else:
            print(f"⚠️ No saved weight file found. Proceeding with initialized state: {best_model_file}")

    model.eval()

    if test_loader is None:
        test_dataset = Dataset(image_dir=img_dir, gt_dir=gt_dir, mode='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    thresholds = np.arange(0.1, 1.0, 0.1)
    sum_tp = {th: 0 for th in thresholds}
    sum_fp = {th: 0 for th in thresholds}
    sum_fn = {th: 0 for th in thresholds}

    with torch.no_grad():
        for batch in tqdm(test_loader):
            norm_image = batch['norm_image'].to(device)
            gt_path = batch['gt_path'][0]
            
            if not os.path.exists(gt_path):
                continue
                
            gt_boundaries = load_gt_boundaries(gt_path)
            
            outputs = model(norm_image)
            edge_output = outputs['edges'].squeeze(0) # [2, H, W]
            
            edge_right = 1.0 - edge_output[0, :, :]
            edge_down = 1.0 - edge_output[1, :, :]

            pred_edge = torch.max(edge_right, edge_down).cpu().numpy() # [H, W]

            for th in thresholds:
                # binarize the precited edge map
                pred_binary = (pred_edge >= th).astype(np.float32)
                
                tps, fps, fns = [], [], []
                for gt in gt_boundaries:
                    # True Positive (predicted 1, ground truth 1)
                    tp = np.sum((pred_binary == 1) & (gt == 1))
                    # False Positive (predicted 1, ground truth 0)
                    fp = np.sum((pred_binary == 1) & (gt == 0))
                    # False Negative (predicted 0, ground truth 1)
                    fn = np.sum((pred_binary == 0) & (gt == 1))
                    
                    tps.append(tp)
                    fps.append(fp)
                    fns.append(fn)
                
                # mean of tps, fps, fns for the current threshold across all ground truths
                sum_tp[th] += np.mean(tps)
                sum_fp[th] += np.mean(fps)
                sum_fn[th] += np.mean(fns)

    best_f1 = -1.0
    best_th = 0.0
    best_prec = 0.0
    best_rec = 0.0

    for th in thresholds:
        tp = sum_tp[th]
        fp = sum_fp[th]
        fn = sum_fn[th]
        
        if tp == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            
        print(f"Threshold: {th:.1f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_prec = precision
            best_rec = recall

    # print("====================================================================")
    # print(f"🏆 Best Threshold: {best_th:.1f}")
    # print(f"✨ Best Precision: {best_prec:.4f}")
    # print(f"✨ Best Recall: {best_rec:.4f}")
    # print(f"🔥 Best F1-Score: {best_f1:.4f}")
    # print("====================================================================")

    return best_f1, best_th, best_prec, best_rec

if __name__ == "__main__":
    evaluate()