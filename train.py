"""드론 탐지 모델 추가 학습 / 중단된 학습 이어하기.

현재 best.pt 는 YOLOv8s 기반이며, 500 epoch 계획 중 21 epoch 에서 멈춘 체크포인트다(mAP50 0.70, 아직 상승 중).
또 AdamW 에 lr0=0.01(SGD 기준 값)을 써서 초반 학습이 흔들렸다(3 epoch 에 mAP50 0.009 로 붕괴).
그래서 기본값은 best.pt 에서 시작해 AdamW lr0=0.001 로 학습하고, 증강 설정은 기존 학습과 같게 둔다.

예)
  python train.py --data dataset/data.yaml                          # best.pt 에서 수정된 설정으로 추가 학습 (권장)
  python train.py --data dataset/data.yaml --model yolov8m.pt       # YOLOv8m 으로 처음부터 학습
  python train.py --resume runs/detect/train-28/weights/last.pt     # 중단된 학습을 기존 설정 그대로 이어서
"""

import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="드론 탐지 모델 학습")
    p.add_argument("--data", help="data.yaml 경로 (--resume 이 아니면 필수)")
    p.add_argument("--model", default="best.pt", help="시작 가중치: best.pt / yolov8s.pt / yolov8m.pt")
    p.add_argument("--resume", help="중단된 학습의 last.pt (기존 학습 설정 그대로 이어감)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50, help="성능 개선이 없으면 조기 종료할 epoch 수")
    p.add_argument("--batch", type=int, default=8, help="-1 이면 GPU 메모리에 맞춰 자동")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=0.001, help="AdamW 는 0.001 전후, SGD 는 0.01")
    p.add_argument("--workers", type=int, default=4, help="Windows 에서 문제가 생기면 0")
    p.add_argument("--device", default=None, help="예: 0, cpu")
    p.add_argument("--name", default=None, help="runs/detect/<name> 에 저장")
    args = p.parse_args()

    if args.resume:
        YOLO(args.resume).train(resume=True)
        return
    if not args.data:
        p.error("--data 가 필요합니다")

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=True,
        degrees=10.0,
        mixup=0.15,
        workers=args.workers,
        device=args.device,
        name=args.name,
    )


if __name__ == "__main__":
    main()
