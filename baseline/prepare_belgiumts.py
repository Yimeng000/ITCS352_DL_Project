import csv
from pathlib import Path
from collections import Counter
from PIL import Image


# ========= 配置区 =========
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
ANNOTATION_DIR = DATA_ROOT / "BelgiumTSD_annotations"
OUTPUT_ROOT = Path(__file__).resolve().parent / "cropped_belgiumts_classid"

TRAIN_FILE = ANNOTATION_DIR / "BTSD_training_GTclear.txt"
TEST_FILE = ANNOTATION_DIR / "BTSD_testing_GTclear.txt"

ALLOWED_CAMERAS = {"00", "01", "02"}

# 这里固定用 class_id
USE_SUPERCLASS = False

IGNORE_SUPERCLASSES = {-1, 0}

MIN_WIDTH = 20
MIN_HEIGHT = 20

# 可选：过滤样本太少的 class_id
# 设为 0 表示不过滤
MIN_SAMPLES_PER_CLASS = 20

SUPERCLASS_NAMES = {
    1: "triangles",
    2: "redcircles",
    3: "bluecircles",
    4: "redbluecircles",
    5: "diamonds",
    6: "revtriangle",
    7: "stop",
    8: "forbidden",
    9: "squares",
    10: "rectanglesup",
    11: "rectanglesdown",
}
# =========================


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def parse_annotation_file(txt_path: Path):
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=";")
        for line_num, parts in enumerate(reader, start=1):
            parts = [p.strip() for p in parts if p.strip() != ""]

            if len(parts) < 7:
                print(f"[WARN] Skip malformed line {line_num} in {txt_path.name}: {parts}")
                continue

            img_rel = parts[0]
            try:
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
                class_id = int(parts[5])
                superclass_id = int(parts[6])
            except ValueError:
                print(f"[WARN] Skip invalid numeric line {line_num} in {txt_path.name}: {parts}")
                continue

            camera = img_rel.split("/")[0]
            image_name = img_rel.split("/")[-1]

            rows.append({
                "camera": camera,
                "img_rel": img_rel,
                "image_name": image_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": class_id,
                "superclass_id": superclass_id,
            })
    return rows


def get_label_name(class_id: int, superclass_id: int):
    if USE_SUPERCLASS:
        if superclass_id in IGNORE_SUPERCLASSES:
            return None
        return SUPERCLASS_NAMES.get(superclass_id, f"superclass_{superclass_id}")

    # class-id 模式下，也要过滤 undefined 类
    if class_id == -1:
        return None

    return f"class_{class_id}"


def crop_one_image(image_path: Path, x1: float, y1: float, x2: float, y2: float):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < MIN_WIDTH or crop_h < MIN_HEIGHT:
        return None
    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))


def collect_class_counts(rows):
    counter = Counter()

    for row in rows:
        if row["camera"] not in ALLOWED_CAMERAS:
            continue

        label_name = get_label_name(row["class_id"], row["superclass_id"])
        if label_name is None:
            continue

        image_path = DATA_ROOT / row["img_rel"]
        if not image_path.exists():
            continue

        # 🔥 加这一行：确保 bbox 也合法
        crop = crop_one_image(
            image_path=image_path,
            x1=row["x1"],
            y1=row["y1"],
            x2=row["x2"],
            y2=row["y2"]
        )

        if crop is None:
            continue

        counter[label_name] += 1

    return counter


def build_allowed_class_set(train_rows, test_rows):
    if MIN_SAMPLES_PER_CLASS <= 0:
        return None

    train_counter = collect_class_counts(train_rows)
    test_counter = collect_class_counts(test_rows)

    allowed = set()
    all_classes = set(train_counter.keys()) | set(test_counter.keys())
    for cls in all_classes:
        # 这里用 train/test 都达到阈值，更稳
        if train_counter.get(cls, 0) >= MIN_SAMPLES_PER_CLASS and test_counter.get(cls, 0) >= MIN_SAMPLES_PER_CLASS:
            allowed.add(cls)

    return allowed


def process_split(rows, split_name: str, allowed_classes=None):
    split_output = OUTPUT_ROOT / split_name
    safe_mkdir(split_output)

    class_counter = Counter()
    skipped_camera = 0
    skipped_label = 0
    skipped_bbox = 0
    skipped_missing = 0
    skipped_rare = 0
    saved_count = 0

    for idx, row in enumerate(rows):
        camera = row["camera"]

        if camera not in ALLOWED_CAMERAS:
            skipped_camera += 1
            continue

        label_name = get_label_name(row["class_id"], row["superclass_id"])
        if label_name is None:
            skipped_label += 1
            continue

        if allowed_classes is not None and label_name not in allowed_classes:
            skipped_rare += 1
            continue

        image_path = DATA_ROOT / row["img_rel"]
        if not image_path.exists():
            skipped_missing += 1
            continue

        crop = crop_one_image(
            image_path=image_path,
            x1=row["x1"],
            y1=row["y1"],
            x2=row["x2"],
            y2=row["y2"]
        )

        if crop is None:
            skipped_bbox += 1
            continue

        class_dir = split_output / label_name
        safe_mkdir(class_dir)

        stem = Path(row["image_name"]).stem
        save_name = (
            f"{camera}_{stem}_"
            f"{int(round(row['x1']))}_{int(round(row['y1']))}_"
            f"{int(round(row['x2']))}_{int(round(row['y2']))}.jpg"
        )
        save_path = class_dir / save_name

        crop.save(save_path, quality=95)
        class_counter[label_name] += 1
        saved_count += 1

        if (idx + 1) % 1000 == 0:
            print(f"[{split_name}] processed {idx + 1}/{len(rows)}")

    print(f"\n===== {split_name.upper()} SUMMARY =====")
    print(f"Saved crops     : {saved_count}")
    print(f"Skipped camera  : {skipped_camera}")
    print(f"Skipped label   : {skipped_label}")
    print(f"Skipped bbox    : {skipped_bbox}")
    print(f"Missing images  : {skipped_missing}")
    print(f"Skipped rare    : {skipped_rare}")
    print("Class counts:")
    for cls, cnt in sorted(class_counter.items(), key=lambda x: x[0]):
        print(f"  {cls:<18} {cnt}")
    print()

    return class_counter


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training annotation not found: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Testing annotation not found: {TEST_FILE}")

    safe_mkdir(OUTPUT_ROOT)

    print("Reading annotations...")
    train_rows = parse_annotation_file(TRAIN_FILE)
    test_rows = parse_annotation_file(TEST_FILE)

    print(f"Train annotations loaded: {len(train_rows)}")
    print(f"Test annotations loaded : {len(test_rows)}\n")

    allowed_classes = build_allowed_class_set(train_rows, test_rows)
    if allowed_classes is not None:
        print(f"Keeping {len(allowed_classes)} class-id categories with >= {MIN_SAMPLES_PER_CLASS} samples in both train and test.\n")

    train_counter = process_split(train_rows, "train", allowed_classes=allowed_classes)
    test_counter = process_split(test_rows, "test", allowed_classes=allowed_classes)

    print("Done.")
    print(f"Output saved to: {OUTPUT_ROOT.resolve()}")

    stats_path = OUTPUT_ROOT / "class_distribution.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("TRAIN\n")
        for cls, cnt in sorted(train_counter.items(), key=lambda x: x[0]):
            f.write(f"{cls}: {cnt}\n")

        f.write("\nTEST\n")
        for cls, cnt in sorted(test_counter.items(), key=lambda x: x[0]):
            f.write(f"{cls}: {cnt}\n")

    print(f"Class distribution saved to: {stats_path.resolve()}")


if __name__ == "__main__":
    main()