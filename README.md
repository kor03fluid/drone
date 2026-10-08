# drone

YOLOv8m으로 드론(회전익 / 고정익)을 탐지하는 프로젝트입니다.
학습 스크립트와 Streamlit 데모 구성은 [JSLEE-0703/Yolov8-drone-detection](https://github.com/JSLEE-0703/Yolov8-drone-detection)을 참고했습니다.

| 항목 | 설정 |
|---|---|
| GPU | NVIDIA RTX 4060 (VRAM 8GB) |
| 모델 | YOLOv8m (`yolov8m.pt`, 첫 실행 때 자동 다운로드) |
| 기본 학습값 | 640px, batch 자동(VRAM 60%), 100 epoch, patience 50, AMP(FP16) |

## 설치

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- `cuXXX`는 [pytorch.org](https://pytorch.org/get-started/locally/)에서 Windows / Pip / CUDA를 골랐을 때 나오는 값을 그대로 쓰세요.
- 그냥 `pip install torch`를 하면 Windows에서는 CPU 버전이 설치됩니다. 마지막 줄이 `True NVIDIA GeForce RTX 4060`으로 나와야 합니다.
- NVIDIA 제어판 → 3D 설정 관리 → **CUDA - Sysmem Fallback Policy**를 **Prefer No Sysmem Fallback**으로 바꾸세요.
  기본값이면 VRAM이 모자랄 때 오류 없이 시스템 RAM을 끌어다 써서 학습이 몇 배 느려집니다.

## 데이터셋

```
dataset/
  train/images/  train/labels/
  valid/images/  valid/labels/
```

클래스는 `data.yaml`에 `0: 회전익`, `1: 고정익`으로 들어 있습니다 (기존 `best.pt`와 같음).
데이터셋에 이미 자체 `data.yaml`이 있다면 `python train.py --data dataset/data.yaml`처럼 지정하면 됩니다.
`test` 항목을 추가하면 학습이 끝난 뒤 test 데이터로도 한 번 더 평가합니다.
OneDrive 폴더 안에 두면 동기화 때문에 데이터 로딩이 느려질 수 있으니 OneDrive 밖에 두는 것을 권장합니다.

## 1. 데이터 점검

```bat
python check_dataset.py
```

- split별 이미지 수, 클래스별 객체 수, 배경 이미지 수를 보여줍니다.
- 학습에서 통째로 빠지게 될 잘못된 라벨(클래스 번호·좌표 오류)을 찾아줍니다.
- 드론이 학습 해상도에서 몇 픽셀로 보이는지 계산해서 `--imgsz`를 추천합니다.
  작은 드론이 많으면 640보다 큰 해상도가 정확도에 가장 크게 영향을 줍니다.

## 2. 학습

```bat
python train.py --imgsz 640
```

`--imgsz`에는 데이터 점검에서 추천받은 값을 넣으세요.
batch는 실제 GPU에서 메모리를 재서 VRAM의 60% 정도를 쓰도록 자동으로 정해지므로, 해상도를 올려도 따로 맞출 필요가 없습니다.
결과는 `runs/detect/yolov8m_drone/`에 저장됩니다 (`weights/best.pt`, `weights/last.pt`, `results.png` 등).

| 상황 | 옵션 |
|---|---|
| batch를 직접 정하고 싶을 때 | `--batch 8` |
| `CUDA out of memory` 오류 | `--batch 4`처럼 더 작게 |
| RAM이 넉넉해서 데이터 로딩을 빠르게 | `--cache ram` |
| 데이터로더(worker) 관련 오류 | `--workers 0` |
| 중간에 멈춘 학습 이어서 하기 | `--resume runs/detect/yolov8m_drone/weights/last.pt` |

학습 로그의 `GPU_mem` 열에 실제 VRAM 사용량이 표시됩니다.

## 3. 데모

```bat
streamlit run app.py
```

가장 최근에 학습한 `runs/detect/*/weights/best.pt`를 자동으로 불러오고, 학습 결과가 없으면 저장소의 `best.pt`를 씁니다.
가중치 경로와 confidence는 사이드바에서 바꿀 수 있습니다.

## 참고

저장소의 `best.pt`는 이전에 YOLOv8s로 학습하던 중간 체크포인트입니다 (500 epoch 중 21 epoch, mAP50 0.70).
