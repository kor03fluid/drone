"""best.pt를 배포용으로 경량화하고 추론 엔진 포맷으로 변환한다.

1) 옵티마이저/EMA 사본 등 학습용 상태를 제거해 weights/<이름>_stripped.pt 로 저장 (원본은 그대로 둠)
2) --format 이 주어지면 그 파일을 ONNX / OpenVINO / TensorRT 등으로 export

예)
  python export.py                              # 경량화만
  python export.py --format openvino            # CPU(인텔)용
  python export.py --format onnx                # 범용 CPU/GPU (onnxruntime)
  python export.py --format engine --half       # NVIDIA GPU (TensorRT FP16)
  python export.py --format openvino --int8 --data path/to/data.yaml   # INT8 양자화 (보정용 데이터 필요)
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer


def main():
    p = argparse.ArgumentParser(description="best.pt 경량화 및 export")
    p.add_argument("--weights", default="best.pt", help="학습 체크포인트 경로")
    p.add_argument("--out-dir", default="weights", help="결과 저장 폴더")
    p.add_argument("--format", choices=["onnx", "openvino", "engine", "torchscript"], help="변환 포맷 (생략 시 경량화만)")
    p.add_argument("--imgsz", type=int, default=640, help="입력 크기 (학습 시 640)")
    p.add_argument("--half", action="store_true", help="FP16 export (GPU/TensorRT, OpenVINO)")
    p.add_argument("--int8", action="store_true", help="INT8 양자화 (--data 필요)")
    p.add_argument("--data", help="INT8 보정에 쓸 data.yaml")
    p.add_argument("--device", default=None, help="예: cpu, 0")
    args = p.parse_args()

    if args.int8 and not args.data:
        p.error("--int8 은 보정용 --data data.yaml 이 필요합니다")

    src = Path(args.weights)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stripped = out_dir / f"{src.stem}_stripped.pt"
    strip_optimizer(src, s=str(stripped))
    print(f"경량화: {src} ({src.stat().st_size / 1e6:.1f} MB) -> {stripped} ({stripped.stat().st_size / 1e6:.1f} MB)")

    if args.format:
        path = YOLO(stripped).export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            data=args.data,
            device=args.device,
        )
        print(f"export 완료: {path}")


if __name__ == "__main__":
    main()
