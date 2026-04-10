from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ALLOWED_CLASSES = [
    "bluecircles",
    "diamonds",
    "rectanglesup",
    "redbluecircles",
    "redcircles",
    "revtriangle",
    "squares",
    "triangles",
]


def get_transforms(img_size: int = 64, augmented: bool = False):
    if augmented:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return train_transform, eval_transform


def _filter_imagefolder_dataset(
    dataset: datasets.ImageFolder,
    allowed_classes: List[str]
) -> Tuple[Subset, List[str]]:
    allowed_classes = sorted(allowed_classes)

    allowed_old_indices = {
        dataset.class_to_idx[class_name]
        for class_name in allowed_classes
        if class_name in dataset.class_to_idx
    }

    filtered_indices = [
        i for i, (_, old_label) in enumerate(dataset.samples)
        if old_label in allowed_old_indices
    ]

    subset = Subset(dataset, filtered_indices)
    return subset, allowed_classes


def _remap_subset_labels(subset: Subset, original_dataset: datasets.ImageFolder, allowed_classes: List[str]):
    new_class_to_idx = {cls_name: i for i, cls_name in enumerate(allowed_classes)}
    old_idx_to_class = {v: k for k, v in original_dataset.class_to_idx.items()}

    remapped_samples = []
    for original_idx in subset.indices:
        path, old_label = original_dataset.samples[original_idx]
        class_name = old_idx_to_class[old_label]
        new_label = new_class_to_idx[class_name]
        remapped_samples.append((path, new_label))

    return remapped_samples, new_class_to_idx


class FilteredImageFolder(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None, classes=None, class_to_idx=None):
        self.samples = samples
        self.transform = transform
        self.classes = classes or []
        self.class_to_idx = class_to_idx or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def extract_group_id(path: str) -> str:
    """
    从文件名提取 group id
    文件名格式:
    00_image.000945_12_34_56_78.jpg

    group id -> 00_image.000945
    """
    filename = Path(path).stem
    parts = filename.split("_")

    if len(parts) < 2:
        return filename

    camera = parts[0]
    stem = parts[1]
    return f"{camera}_{stem}"


def grouped_train_val_split(samples, val_ratio=0.2, seed=42):
    """
    按 group 分 train / val
    同一个 group 的样本不会同时出现在 train 和 val
    """
    group_to_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, (path, _) in enumerate(samples):
        group_id = extract_group_id(path)
        group_to_indices[group_id].append(idx)

    group_ids = list(group_to_indices.keys())

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(group_ids), generator=generator).tolist()
    group_ids = [group_ids[i] for i in perm]

    total_samples = len(samples)
    target_val_size = int(total_samples * val_ratio)

    train_indices = []
    val_indices = []
    val_count = 0

    for gid in group_ids:
        group_indices = group_to_indices[gid]

        if val_count < target_val_size:
            val_indices.extend(group_indices)
            val_count += len(group_indices)
        else:
            train_indices.extend(group_indices)

    return train_indices, val_indices


def get_dataloaders(
    data_root: str = "cropped_belgiumts",
    batch_size: int = 32,
    img_size: int = 64,
    val_ratio: float = 0.2,
    augmented: bool = False,
    num_workers: int = 2,
):
    data_root = Path(data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_transform, eval_transform = get_transforms(img_size=img_size, augmented=augmented)

    raw_train_dataset = datasets.ImageFolder(root=str(train_dir), allow_empty=True)
    raw_test_dataset = datasets.ImageFolder(root=str(test_dir), allow_empty=True)

    train_subset, allowed_classes = _filter_imagefolder_dataset(raw_train_dataset, ALLOWED_CLASSES)
    test_subset, _ = _filter_imagefolder_dataset(raw_test_dataset, ALLOWED_CLASSES)

    train_samples, class_to_idx = _remap_subset_labels(train_subset, raw_train_dataset, allowed_classes)
    test_samples, _ = _remap_subset_labels(test_subset, raw_test_dataset, allowed_classes)

    full_train_dataset = FilteredImageFolder(
        samples=train_samples,
        transform=train_transform,
        classes=allowed_classes,
        class_to_idx=class_to_idx,
    )

    full_train_eval_dataset = FilteredImageFolder(
        samples=train_samples,
        transform=eval_transform,
        classes=allowed_classes,
        class_to_idx=class_to_idx,
    )

    test_dataset = FilteredImageFolder(
        samples=test_samples,
        transform=eval_transform,
        classes=allowed_classes,
        class_to_idx=class_to_idx,
    )

    train_indices, val_indices = grouped_train_val_split(
        train_samples,
        val_ratio=val_ratio,
        seed=42
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_eval_dataset, val_indices)

    print(f"Total train samples: {len(train_samples)}")
    print(f"Grouped train samples: {len(train_indices)}")
    print(f"Grouped val samples: {len(val_indices)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, allowed_classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))