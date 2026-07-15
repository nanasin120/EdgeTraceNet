# EdgeTraceNet: Get an edge by self-supervised learning

EdgeTraceNet은 정답 라벨(Ground Truth) 없이 이미지의 픽셀 간 연결성을 분석하여 사물의 윤곽선(Edge)을 스스로 학습하는 자기지도학습(Self-Supervised Learning) 모델입니다.

---

## 1. 주요 특징 (Key Features)

- **비지도/자기지도 학습 기반:** 오직 이미지 데이터만을 이용해 윤곽선을 추출해냅니다.
- **ResNet-18 백본 활용:** 대표적인 CNN 아키텍처인 ResNet-18을 인코더로 채택하여 가볍고 빠른 추론 속도(High FPS)를 보장합니다.
- **특징을 이용한 손실:** 특징간의 유사도를 손실함수에 이용해 색상만으로 파악하지 못하는 부분을 채웁니다.

---

## 2. 개발 환경 및 하드웨어 (Environment & Hardware)

본 프로젝트는 아래의 하드웨어 및 소프트웨어 환경에서 테스트 및 개발되었습니다. (Google Colab 환경에서도 학습 및 구동이 가능합니다.)

| Component | Specification |
| :--- | :--- |
| **Laptop Model** | MSI Alpha 17 C7VG |
| **GPU** | NVIDIA GeForce RTX 4070 Laptop (8GB VRAM) |
| **OS** | Windows 11 |
| **CUDA / Driver** | CUDA 13.1 / Driver 591.74 |

---

## 3. 모델 아키텍처 및 파이프라인 (Architecture & Pipeline)

### 학습 아키텍처 (Input Structure)
- **이미지 색상 및 특징:** 픽셀간의 연결성(오른쪽과 아래쪽)을 윤곽선으로 이용해 색상 및 특징간의 유사도를 가르는데 이용합니다.

### 손실 함수 (Loss Functions) & 최적화 (Optimization)
- **Loss Composition**
  - Color Difference Loss (Weight: 1.0)
  - Feature Difference Loss (Weight: 1.0)
  - Regularization Loss (Weight: 0.01)
  - Binarization Loss (Weight: 0.001)
  - Thick Edge Loss (Weight: 0.1)

- **Optimizer & Scheduler:** `AdamW` (Initial LR: 1e-4) + `CosineAnnealingLR` (Min LR: 1e-6)

---

## 4. 시작하기 (Getting Started)

### 4-1. Installation

이 프로젝트는 Anaconda 가상환경을 사용하여 원활하게 실행할 수 있도록 구성되어 있습니다. 아래 명령어를 순서대로 실행하여 환경을 구축해 주세요.

**Repository 복제:**
 ```bash
 git clone https://github.com/nanasin120/EdgeTraceNet.git
 cd EdgeTraceNet
 ```

Conda 가상환경 생성 및 활성화:
  ```Bash
  conda create -n edgetrace python=3.10 -y
  conda activate edgetrace
  ```

필수 라이브러리 설치:
PyTorch는 사용하시는 GPU 및 CUDA 버전에 맞게 설치해주시기 바랍니다.
  ```Bash
  # PyTorch 설치 예시 (CUDA 11.8)
  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
  
  # 기타 필수 종속성 설치
  pip install numpy scipy pillow tqdm
  ```

### 4-2. How to Run (실행 방법)

설치와 데이터셋 준비가 완료되었다면, 아래 명령어를 통해 학습 및 검증(평가)을 진행할 수 있습니다.

#### 1. 모델 학습 (Training)
학습을 시작하려면 아래 명령어를 실행해 주세요. 주요 하이퍼파라미터(에폭 수, 배치 크기, 손실 함수 가중치 등)는 명령행 인자(Arguments)로 자유롭게 조정할 수 있습니다.

```bash
python train.py \
  --img_dir "./data/BSDS500/images" \
  --gt_dir "./data/BSDS500/ground_truth" \
  --model_save_path "./save/model_save" \
  --img_save_path "./save/image_save" \
  --batch_size 16 \
  --epochs 500
```

#### 2. 모델 평가 및 추론 (Evaluation)
학습된 최적의 체크포인트 가중치(best_model_epoch.pth)를 불러와 테스트 데이터셋에 대해 Precision, Recall, F1-Score를 측정하고 시각화 결과를 저장합니다.

```Bash
python eval.py
```

#### 3. 주요 설정 옵션 (Arguments)
학습 시 사용 가능한 주요 옵션과 기본값(Default)은 다음과 같습니다. 다른 설정으로 실험하고 싶다면 실행 명령어 뒤에 인자를 추가해 주세요.

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--epochs` | `int` | `500` | 총 학습 에폭 수 |
| `--batch_size` | `int` | `16` | 학습 배치 크기 |
| `--lr` | `float` | `1e-4` | 초기 학습률 (Learning Rate) |
| `--weight_color` | `float` | `1.0` | Color Difference Loss 가중치 |
| `--weight_feature` | `float` | `1.0` | Feature Difference Loss 가중치 |
| `--weight_thick` | `float` | `0.1` | Thick Edge Loss 가중치 (선 두께 규제) |
| `--image_save_interval` | `int` | `10` | 시각화 결과(`vis_epoch_X.png`) 저장 에폭 주기 |
| `--weight_save_interval` | `int` | `50` | 모델 가중치 저장 에폭 주기 |

* **실행 예시 (학습률과 가중치를 조정하여 실험할 때):**
  ```bash
  python train.py --lr 5e-5 --weight_thick 0.5 --batch_size 8
  ```
  
---

## 5. 평가 지표 및 결과 (Evaluation & Results)

### 5-1. 정량적 평가 지표 (Evaluation Metrics)

| Method / Stage | Best Threshold | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |  :---: |
| **EdgeTraceNet (Initial)** | 0.000 | 0.000 | 0.000 | 0.000 |
| **EdgeTraceNet (100)** | 0.000 | 0.000 | 0.000 | 0.000 |
| **EdgeTraceNet (200)** | 0.000 | 0.000 | 0.000 | 0.000 |
| **EdgeTraceNet (300)** | 0.000 | 0.000 | 0.000 | 0.000 |
| **EdgeTraceNet (400)** | 0.000 | 0.000 | 0.000 | 0.000 |
| **EdgeTraceNet (500)** | 0.000 | 0.000 | 0.000 | 0.000 |

### 5-2. 결과 (Results)

(추후 500 에폭 최종 완주 후 시각화 결과 이미지 추가 예정)

---

## 안내 및 출처 (Acknowledgements)

본 프로젝트는 캐글(Kaggle)을 통해 제공된 Berkeley Segmentation Dataset의 수정 버전을 사용하였습니다. 데이터셋 출처의 요청에 따라, 본 프로젝트를 인용하실 때는 아래의 연구 논문을 함께 인용해 주시기 바랍니다:

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

