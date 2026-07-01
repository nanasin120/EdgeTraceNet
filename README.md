# EdgeTraceNet: Get an edge by self-supervised learning

EdgeTraceNet은 정답 라벨(Ground Truth) 없이 이미지의 픽셀 간 연결성을 분석하여 사물의 윤곽선(Edge)을 스스로 학습하는 자기지도학습(Self-Supervised Learning) 모델입니다.

## Overview
이 프로젝트는 전통적인 컴퓨터 비전의 '연결성(Affinity)' 개념을 현대적인 딥러닝 아키텍처와 결합했습니다. 복잡한 라벨링 과정 없이, 모델 스스로 이미지 내부의 색상 차이를 분석하여 사물의 경계를 찾아냅니다.

- Self-Supervised: 외부 정답지 없이 오직 이미지 자체의 구조적 특징만으로 학습합니다.
- Connectivity-based: 픽셀 간의 관계를 수치화하여 윤곽선을 정의합니다.
- Lightweight: 이미지 한 장으로도 강력한 최적화가 가능합니다.

## Results (40 Epochs)

40회 반복시 결과입니다.

<img width="448" height="224" alt="image" src="https://github.com/user-attachments/assets/4fb7e9b1-60e7-4af6-b4d3-ee233c52ae7b" />


## How it works?

EdgeTraceNet은 두 가지 핵심 손실 함수(Loss Function)의 조화로 학습됩니다.

Color Difference Loss: 색상 차이가 큰 곳은 '경계'로 판단하여 연결을 끊도록 유도합니다.

Regularization Loss: 꼼수(전부 끊어버리기)를 방지하고, 가능한 한 인접 픽셀들을 연결하도록 유도합니다.

## License
MIT License를 사용하여 자유롭게 사용 및 수정이 가능합니다.
