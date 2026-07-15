import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="EdgeTraceNet Training Script")

    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch number')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--weight_color', type=float, default=1.0, help='weight of color loss')
    parser.add_argument('--weight_feature', type=float, default=1.0, help='weight of feature loss')
    parser.add_argument('--weight_regular', type=float, default=0.001, help='weight of regular loss')
    parser.add_argument('--weight_binary', type=float, default=0.001, help='weight of binary loss')
    parser.add_argument('--weight_thick', type=float, default=0.1, help='weight of thick loss')
    
    parser.add_argument('--image_save_interval', type=int, default=10, help='how many epochs to wait before saving training images')
    parser.add_argument('--weight_save_interval', type=int, default=50, help='how many epochs to wait before saving model weights')
    
    parser.add_argument('--img_dir', type=str, default=r"data/BSDS500/images", help='path to images')
    parser.add_argument('--gt_dir', type=str, default=r"data/BSDS500/ground_truth", help='path to ground truthes')

    parser.add_argument('--model_save_path', type=str, default=os.path.normpath('./save/model_save'), help='path to save checkpoints')
    parser.add_argument('--img_save_path', type=str, default=os.path.normpath('./save/image_save'), help='path to save generated images')
    
    return parser.parse_args()