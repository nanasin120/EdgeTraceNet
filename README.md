# EdgeTraceNet: Get an edge by self-supervised learning

EdgeTraceNet is a self-supervised learning model that learns the edges of objects by itself by analyzing the connectivity between pixels of an image without ground truth labels.

---

## 1. Key Features

- **Based on Unsupervised/Self-Supervised Learning:** Extracts edges using only image data.
- **Using ResNet-18 Backbone:** Adopts ResNet-18, a representative CNN architecture, as an encoder to guarantee lightweight and fast inference speed (High FPS).
- **Loss using Features:** Uses the similarity between features in the loss function to fill in the parts that cannot be identified by color alone.

---

## 2. Environment & Hardware

This project was tested and developed in the hardware and software environment below. (Learning and running are also possible in the Google Colab environment.)

| Component | Specification |
| :--- | :--- |
| **Laptop Model** | MSI Alpha 17 C7VG |
| **GPU** | NVIDIA GeForce RTX 4070 Laptop (8GB VRAM) |
| **OS** | Windows 11 |
| **CUDA / Driver** | CUDA 13.1 / Driver 591.74 |

---

## 3. Architecture & Pipeline

### Input Structure
- **Image Color and Features:** Uses the connectivity between pixels (right and bottom) as edges, and uses it to separate the similarity between colors and features.

### Loss Functions & Optimization
- **Loss Composition**
  - Color Difference Loss (Weight: 1.0)
  - Feature Difference Loss (Weight: 1.0)
  - Regularization Loss (Weight: 0.01)
  - Binarization Loss (Weight: 0.001)
  - Thick Edge Loss (Weight: 0.1)

- **Optimizer & Scheduler:** `AdamW` (Initial LR: 1e-4) + `CosineAnnealingLR` (Min LR: 1e-6)

---

## 4. Getting Started

### 4-1. Installation

This project is configured so that it can be executed smoothly using an Anaconda virtual environment. Please run the commands below in order to build the environment.

**Clone Repository:**
 ```bash
 git clone https://github.com/nanasin120/EdgeTraceNet.git
 cd EdgeTraceNet
 ```

Create and Activate Conda Virtual Environment:
  ```Bash
  conda create -n edgetrace python=3.10 -y
  conda activate edgetrace
  ```

Install Required Libraries:
Please install PyTorch according to the GPU and CUDA version you use.
  ```Bash
  # PyTorch Installation Example (CUDA 11.8)
  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
   
  # Install other required dependencies
  pip install numpy scipy pillow tqdm
  ```

### 4-2. How to Run

If the installation and dataset preparation are completed, you can proceed with training and validation (evaluation) through the commands below.

#### 1. Training
To start training, please run the command below. Major hyperparameters (number of epochs, batch size, loss function weights, etc.) can be freely adjusted with command line arguments.

```bash
python train.py \
  --img_dir "./data/BSDS500/images" \
  --gt_dir "./data/BSDS500/ground_truth" \
  --model_save_path "./save/model_save" \
  --img_save_path "./save/image_save" \
  --batch_size 16 \
  --epochs 500
```

#### 2. Evaluation and Inference
Loads the best trained checkpoint weight (best_model_epoch.pth) to measure Precision, Recall, and F1-Score for the test dataset and save the visualization results.

```Bash
python eval.py
```

#### 3. Major Configuration Options (Arguments)
Major options and default values available during training are as follows. If you want to experiment with other configurations, please add arguments after the execution command.

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--epochs` | `int` | `500` | Total number of training epochs |
| `--batch_size` | `int` | `16` | Training batch size |
| `--lr` | `float` | `1e-4` | Initial learning rate |
| `--weight_color` | `float` | `1.0` | Weight for Color Difference Loss |
| `--weight_feature` | `float` | `1.0` | Weight for Feature Difference Loss |
| `--weight_thick` | `float` | `0.1` | Weight for Thick Edge Loss (line thickness regulation) |
| `--image_save_interval` | `int` | `10` | Saving epoch interval for visualization results (vis_epoch_X.png) |
| `--weight_save_interval` | `int` | `50` | Saving epoch interval for model weights |

* **Execution Example (When experimenting by adjusting learning rate and weights):**
  ```bash  
  python train.py --lr 5e-5 --weight_thick 0.5 --batch_size 8
  ```
  
---

## 5. Evaluation & Results

### 5-1. Evaluation Metrics

| Method / Stage | Best Threshold | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |  :---: |
| **EdgeTraceNet (Initial)** | 0.1 | 0.0187 | 1.000 | 0.0367 |
| **EdgeTraceNet (100)** | 0.9 | 0.0466 | 0.2118 | 0.0764 |
| **EdgeTraceNet (200)** | 0.9 | 0.0591 | 0.1834 | 0.0894 |
| **EdgeTraceNet (300)** | 0.9 | 0.0547 | 0.1899 | 0.0849 |
| **EdgeTraceNet (400)** | 0.9 | 0.0526 | 0.1943 | 0.0828 |
| **EdgeTraceNet (500)** | 0.9 | 0.0531 | 0.1929 | 0.0833 |

### 5-2. Results

They are 0 epoch, 100 epoch, 200 epoch, 300 epoch, 400 epoch, and 500 epoch in order.

<img width="1292" height="1300" alt="ezgif com-gif-maker" src="https://github.com/user-attachments/assets/d033fe67-63b3-4b15-add3-b7cc19cb9bcc" />

---

## Acknowledgements

This project used a modified version of the Berkeley Segmentation Dataset provided through Kaggle. According to the request of the dataset source, please cite the research paper below when citing this project:

```bibtex
@InProceedings{MartinFTM01,
  author = {D. Martin and C. Fowlkes and D. Tal and J. Malik},
  title = {A Database of Human Segmented Natural Images and its
           Application to Evaluating Segmentation Algorithms and
           Measuring Ecological Statistics},
  booktitle = {Proc. 8th Int'l Conf. Computer Vision},
  year = {2001},
  month = {July},
  volume = {2},
  pages = {416--423}
}

