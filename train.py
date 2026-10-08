"""YOLOv8m 드론 탐지 학습 스크립트 (RTX 4060 8GB 기준).

사용 예:
    python check_dataset.py                  # 먼저 데이터 점검 + imgsz 추천
    python train.py                          # 640px, batch는 VRAM에 맞춰 자동
    python train.py --imgsz 960              # 작은 드론이 많을 때
    python train.py --batch 8                # batch 직접 지정
    python train.py --cache ram              # RAM이 넉넉하면 데이터 로딩 가속
    python train.py --resume runs/detect/yolov8m_drone/weights/last.pt
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def batch_size(value):
    """정수는 배치 크기, -1은 AutoBatch(VRAM 60%), 0~1 실수는 AutoBatch가 쓸 VRAM 비율."""
    v = float(value)
    return int(v) if v.is_integer() else v


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8m drone detection training")
    p.add_argument("--model", default="yolov8m.pt", help="사전학습 가중치 (처음 실행 시 자동 다운로드)")
    p.add_argument("--data", default=str(ROOT / "data.yaml"), help="데이터셋 설정 파일")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=50, help="검증 성능이 이 epoch 수만큼 안 오르면 조기 종료")
    p.add_argument("--imgsz", type=int, default=640)
    # imgsz를 바꿔도 VRAM 8GB를 넘지 않도록 실제 GPU에서 메모리를 재서 batch를 정함
    p.add_argument("--batch", type=batch_size, default=-1, help="기본값 -1: VRAM의 60%%를 쓰도록 자동 결정")
    p.add_argument("--workers", type=int, default=4, help="데이터로더 프로세스 수")
    p.add_argument("--device", default="0", help="GPU 번호, CPU는 'cpu'")
    p.add_argument("--cache", choices=["none", "ram", "disk"], default="none", help="이미지 캐시 위치")
    p.add_argument("--name", default="yolov8m_drone", help="runs/detect/ 아래 결과 폴더 이름")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", metavar="LAST_PT", help="중단된 학습을 이어서 할 last.pt 경로")
    return p.parse_args()


def check_gpu(device):
    if device == "cpu":
        print("경고: CPU로 학습합니다. YOLOv8m은 CPU에서 매우 느립니다.")
        return
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU를 찾을 수 없습니다. CUDA용 PyTorch가 설치됐는지 확인하세요 (README의 설치 참고).\n"
            "CPU로 강제 실행하려면 --device cpu 를 붙이세요."
        )
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB), PyTorch {torch.__version__}")


def evaluate_test(trainer):
    """data.yaml에 test 항목이 있으면 학습·모델 선택에 쓰지 않은 test 데이터로 최종 성능을 측정."""
    if not trainer.data.get("test"):
        return
    print("\ntest 데이터로 최종 평가")
    YOLO(trainer.best).val(
        data=trainer.args.data,
        split="test",
        imgsz=trainer.args.imgsz,
        batch=trainer.batch_size * 2,  # 학습 중 검증과 같은 크기
        device=trainer.args.device,
        project=str(trainer.save_dir.parent),
        name=f"{trainer.save_dir.name}-test",
    )


def main():
    args = parse_args()
    check_gpu(args.device)

    if args.resume:
        model = YOLO(args.resume)
        # 끝난 학습의 체크포인트를 넘기면 Ultralytics가 경고만 하고 기본 데이터셋(coco8)으로 새 학습을 시작하므로 미리 막음
        ckpt = model.ckpt or {}
        if ckpt.get("epoch", -1) < 0 or ckpt.get("optimizer") is None:
            raise SystemExit(
                f"{args.resume} 은(는) 이어서 학습할 수 없는 체크포인트입니다 (학습이 끝나 optimizer 상태가 없음).\n"
                "중간에 멈춘 학습의 weights/last.pt 를 지정하세요."
            )
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            cache=False if args.cache == "none" else args.cache,
            seed=args.seed,
            # 절대경로로 지정해야 Ultralytics 전역 설정(runs_dir)과 무관하게 이 폴더 아래에 저장됨
            project=str(ROOT / "runs" / "detect"),
            name=args.name,
        )

    evaluate_test(model.trainer)
    print(f"\n학습 완료. best 가중치: {model.trainer.best}")


if __name__ == "__main__":  # Windows에서 데이터로더 멀티프로세싱에 필요
    main()
