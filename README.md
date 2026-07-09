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
*(추후 업데이트 예정)*

### 4-2. Dataset Preparation
*(추후 업데이트 예정)*

---

## 5. 평가 지표 및 결과 (Evaluation & Results)

### 5-1. 정량적 평가 지표 (Evaluation Metrics)

| Method | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **EdgeTraceNet (Ours)** | 0.000 | 0.000 | 0.000 |

### 5-2. 결과 (Results)

500 에포크를 돌린 결과 입니다.

<img width="908" height="916" alt="vis_epoch_500" src="https://github.com/user-attachments/assets/ab35ac34-f2ef-42de-a57d-27954fed6b24" />

**설명:** 

나쁘지는 않지만 보이지 않는 무언가의 윤곽선이 잡히는 모습이 보입니다.

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

