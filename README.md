# drone

YOLOv8 드론 탐지 모델(`best.pt`)과 추론·경량화·학습 스크립트.

## 모델 정보 (`best.pt` 체크포인트 기준)

| 항목 | 값 |
|---|---|
| 아키텍처 | **YOLOv8s** (depth 0.33 / width 0.50, 파라미터 11.1M, 시작 가중치 `yolov8s.pt`) |
| 클래스 | `0: 회전익`, `1: 고정익` |
| 입력 크기 | 640 |
| 학습 진행 | 500 epoch 계획 중 **21 epoch** 에서 저장된 체크포인트 (옵티마이저 상태 포함, 89.5 MB) |
| 검증 성능 | P 0.663 / R 0.600 / mAP50 0.702 / mAP50-95 0.374 |

학습이 끝나기 전에 멈춘 체크포인트라 성능이 아직 오르는 중이었다. 또 AdamW 에 `lr0=0.01`(SGD 기준 값)을 써서 초반이 불안정했다(3 epoch 에 mAP50 0.009). `train.py` 는 이 둘을 바로잡은 설정이다.

## 설치

```bash
pip install ultralytics
```

## 사용법

### 탐지 / 추적

```bash
python detect.py video.mp4                      # 결과는 runs/detect/predict 에 저장
python detect.py video.mp4 --track              # ID 부여 추적 (ByteTrack)
python detect.py 0 --show                       # 웹캠
python detect.py video.mp4 --agnostic-nms       # 한 물체에 회전익·고정익 박스가 같이 뜨는 것 억제
```

### 경량화 / 변환

```bash
python export.py                                # 옵티마이저 제거: 89.5 MB -> 22.5 MB (weights/best_stripped.pt)
python export.py --format openvino              # CPU 추론용 (권장)
python export.py --format engine --half         # NVIDIA GPU 용 TensorRT FP16
python export.py --format openvino --int8 --data dataset/data.yaml   # INT8 (보정용 데이터 필요)
```

변환한 모델도 `detect.py --weights weights/best_stripped_openvino_model` 처럼 그대로 쓴다.

CPU 4코어, 1080p 영상, imgsz 640 에서 측정한 추론 시간:

| 포맷 | 추론 (ms/프레임) |
|---|---|
| PyTorch `best.pt` | 73.5 |
| ONNX Runtime | 83.6 |
| **OpenVINO** | **32.7** (약 2.2배) |

### 학습

```bash
python train.py --data dataset/data.yaml                        # best.pt 에서 AdamW lr0=0.001 로 추가 학습 (권장)
python train.py --data dataset/data.yaml --model yolov8m.pt     # YOLOv8m 으로 처음부터
python train.py --resume runs/detect/train-28/weights/last.pt   # 중단된 학습을 기존 설정 그대로 이어서
```
