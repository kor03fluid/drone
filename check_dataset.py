"""데이터셋 점검: 이미지·라벨 수, 클래스 분포, 라벨 오류, 객체 크기를 확인하고 학습 이미지 크기(imgsz)를 추천합니다.

사용 예:
    python check_dataset.py
    python check_dataset.py --data dataset/data.yaml
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

ROOT = Path(__file__).resolve().parent
IMGSZ_CANDIDATES = (640, 800, 960, 1280)
TINY_PX = 10  # YOLOv8의 가장 촘촘한 출력 격자 간격(8px)과 비슷한 크기. 이보다 작은 객체는 놓치기 쉬움
MAX_TINY_RATIO = 0.10  # 아주 작은 객체 비율을 이 이하로 만드는 가장 작은 imgsz를 추천


def parse_args():
    p = argparse.ArgumentParser(description="데이터셋 점검 및 imgsz 추천")
    p.add_argument("--data", default=str(ROOT / "data.yaml"), help="데이터셋 설정 파일")
    return p.parse_args()


def image_files(source):
    """data.yaml의 train/val/test 항목(폴더, 이미지 목록 txt, 또는 그 리스트)에서 이미지 경로를 모음."""
    files = []
    for s in source if isinstance(source, list) else [source]:
        p = Path(s)
        if p.is_dir():
            files += [f for f in p.rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]
        elif p.is_file():
            for line in p.read_text(encoding="utf-8").split():
                f = Path(line) if Path(line).is_absolute() else p.parent / line
                if f.suffix[1:].lower() in IMG_FORMATS:
                    files.append(f)
    return sorted(files)


def read_label(path, nc):
    """Ultralytics와 같은 기준으로 라벨 파일을 검사해 [(클래스, w, h), ...]를 반환. 문제가 있으면 ValueError."""
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        v = np.array(line.split(), dtype=float)
        if len(v) == 5:
            xy, w, h = v[1:], v[3], v[4]
        elif len(v) > 6 and len(v) % 2 == 1:  # 폴리곤 라벨은 학습 때 외접 박스로 바뀜
            xy = v[1:]
            w, h = np.ptp(xy.reshape(-1, 2), axis=0)
        else:
            raise ValueError(f"한 줄의 값 개수가 {len(v)}개 (박스는 5개여야 함)")
        if xy.max() > 1.01 or v.min() < -0.01:
            raise ValueError("좌표가 0~1 범위를 벗어남 (정규화 안 된 좌표)")
        if v[0] % 1 or v[0] >= nc:
            raise ValueError(f"클래스 번호 {v[0]:g}가 0~{nc - 1} 범위가 아님")
        boxes.append((int(v[0]), w, h))
    return boxes


def check_split(split, source, names):
    """split 하나를 점검해 출력하고, 객체 크기(이미지 긴 변 대비 비율)와 이미지 긴 변 길이를 반환."""
    imgs = image_files(source)
    counts, background, bad = Counter(), 0, []
    rel_sizes, long_sides = [], []
    for img, lb in zip(imgs, img2label_paths([str(f) for f in imgs])):
        lb = Path(lb)
        try:
            boxes = read_label(lb, len(names)) if lb.is_file() else []
            with Image.open(img) as im:
                w_img, h_img = im.size
        except (ValueError, OSError) as e:
            bad.append(f"{lb if isinstance(e, ValueError) else img}: {e}")
            continue
        if not boxes:
            background += 1
            continue
        counts.update(c for c, _, _ in boxes)
        long_side = max(w_img, h_img)
        long_sides.append(long_side)
        rel_sizes += [np.sqrt(w * w_img * h * h_img) / long_side for _, w, h in boxes]

    print(f"\n[{split}] 이미지 {len(imgs)}장 | 객체 있음 {len(imgs) - background - len(bad)} | 배경 {background} | 오류 {len(bad)}")
    print("  " + " | ".join(f"{name} {counts[i]}개" for i, name in names.items()))
    if bad:
        print(f"  오류가 있는 이미지 {len(bad)}장은 학습에서 통째로 제외됩니다. 예:")
        for msg in bad[:5]:
            print(f"    {msg}")
    present = [counts[i] for i in names]
    if min(present) == 0:
        print("  경고: 객체가 하나도 없는 클래스가 있습니다.")
    elif max(present) / min(present) >= 5:
        print("  참고: 클래스 간 객체 수 차이가 5배 이상입니다. 적은 클래스의 성능이 낮게 나올 수 있습니다.")
    return rel_sizes, long_sides


def recommend_imgsz(rel_sizes, long_sides):
    rel_sizes = np.array(rel_sizes)
    native = float(np.median(long_sides))
    # 원본보다 크게 키워도 정보가 늘지 않으므로 원본 해상도까지만 후보로 씀
    candidates = [s for s in IMGSZ_CANDIDATES if s <= max(native, IMGSZ_CANDIDATES[0])]

    print(f"\n학습 이미지 크기(imgsz)별 객체 크기 (train, √(w·h) 픽셀) / 원본 긴 변 중앙값 {native:.0f}px")
    print(f"  imgsz   중앙값   하위10%  {TINY_PX}px 미만")  # 한글은 두 칸 폭이라 직접 맞춤
    tiny = {}
    for s in candidates:
        px = rel_sizes * s
        tiny[s] = float((px < TINY_PX).mean())
        print(f"  {s:>5}  {np.median(px):>7.1f}  {np.percentile(px, 10):>8.1f}  {tiny[s]:>9.1%}")

    rec = next((s for s in candidates if tiny[s] <= MAX_TINY_RATIO), candidates[-1])
    print(f"\n추천: python train.py --imgsz {rec}")
    if rec > IMGSZ_CANDIDATES[0]:
        print(f"  640 대비 학습 시간·VRAM이 약 {(rec / 640) ** 2:.1f}배 듭니다. batch는 train.py가 VRAM에 맞춰 자동으로 정합니다.")
    if tiny[rec] > MAX_TINY_RATIO:
        print(f"  imgsz {rec}에서도 {TINY_PX}px 미만 객체가 {tiny[rec]:.0%}입니다. 아주 작은 드론은 놓칠 수 있습니다.")


def main():
    args = parse_args()
    try:
        data = check_det_dataset(args.data, autodownload=False)
    except Exception as e:
        raise SystemExit(f"데이터셋을 읽을 수 없습니다: {e}")

    print(f"클래스: {data['names']}")
    stats = {split: check_split(split, data[split], data["names"]) for split in ("train", "val", "test") if data.get(split)}
    rel_sizes, long_sides = stats["train"]
    if not rel_sizes:
        raise SystemExit("\ntrain에 사용할 수 있는 라벨이 없습니다.")
    recommend_imgsz(rel_sizes, long_sides)


if __name__ == "__main__":
    main()
