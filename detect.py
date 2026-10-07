"""이미지/영상/폴더/웹캠에서 드론(회전익·고정익)을 탐지하거나 추적한다.

.pt 뿐 아니라 export.py 로 만든 .onnx, *_openvino_model/, .engine 도 --weights 로 그대로 쓸 수 있다.

예)
  python detect.py video.mp4
  python detect.py video.mp4 --track                                   # ID 부여 추적
  python detect.py 0 --show                                            # 웹캠
  python detect.py video.mp4 --weights weights/best_stripped_openvino_model
"""

import argparse
from collections import Counter

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="드론 탐지/추적")
    p.add_argument("source", help="이미지·영상·폴더 경로, RTSP URL, 웹캠은 0")
    p.add_argument("--weights", default="best.pt", help=".pt / .onnx / *_openvino_model / .engine")
    p.add_argument("--imgsz", type=int, default=640, help="입력 크기 (학습 시 640, 작은 드론이 많으면 960~1280 시도)")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU 임계값")
    p.add_argument("--agnostic-nms", action="store_true", help="같은 물체에 회전익·고정익 박스가 겹쳐 나오면 하나만 남김")
    p.add_argument("--track", action="store_true", help="객체 추적(ID 부여)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml(빠름) / botsort.yaml")
    p.add_argument("--vid-stride", type=int, default=1, help="영상에서 N프레임마다 1번 추론")
    p.add_argument("--half", action="store_true", help="FP16 추론 (CUDA GPU)")
    p.add_argument("--device", default=None, help="예: cpu, 0")
    p.add_argument("--show", action="store_true", help="결과 창 표시")
    p.add_argument("--no-save", action="store_true", help="결과 이미지/영상 저장 안 함")
    args = p.parse_args()

    model = YOLO(args.weights, task="detect")
    kwargs = dict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        vid_stride=args.vid_stride,
        half=args.half,
        device=args.device,
        show=args.show,
        save=not args.no_save,
        stream=True,
        verbose=False,
    )
    results = model.track(tracker=args.tracker, persist=True, **kwargs) if args.track else model.predict(**kwargs)

    frames = hit_frames = 0
    counts = Counter()
    track_ids = set()
    speed = Counter()
    save_dir = None
    for r in results:
        frames += 1
        save_dir = r.save_dir
        speed.update(r.speed)
        if len(r.boxes):
            hit_frames += 1
        counts.update(r.names[int(c)] for c in r.boxes.cls.tolist())
        if r.boxes.id is not None:
            track_ids.update(int(i) for i in r.boxes.id.tolist())

    if not frames:
        print("입력에서 프레임을 읽지 못했습니다")
        return
    print(f"프레임 {frames}개 중 {hit_frames}개에서 탐지 ({hit_frames / frames:.0%}), 클래스별 박스 수: {dict(counts)}")
    if args.track:
        print(f"추적된 고유 ID 수: {len(track_ids)}")
    print(
        "평균 처리 시간(ms/프레임): "
        + ", ".join(f"{k} {v / frames:.1f}" for k, v in speed.items())
    )
    if not args.no_save:
        print(f"결과 저장 위치: {save_dir}")


if __name__ == "__main__":
    main()
