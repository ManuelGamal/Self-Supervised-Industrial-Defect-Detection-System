"""
Dataset classes for industrial defect detection.
Supports MVTec AD, DAGM, and NEU Surface Defect datasets.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class IndustrialDataset(Dataset):
    """
    Generic industrial image dataset.
    Supports normal-only (SSL) and labeled (fine-tuning) modes.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        mode: str = "ssl",  # 'ssl' or 'supervised'
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.mode = mode
        self.transform = transform
        self.image_size = image_size

        self.image_paths, self.labels = self._load_file_list()

    def _load_file_list(self) -> Tuple[List[Path], List[int]]:
        """Load image paths and labels from split directory."""
        split_dir = self.root_dir / self.split
        image_paths = []
        labels = []

        if self.mode == "ssl":
            # Only normal images for self-supervised pretraining
            normal_dir = split_dir / "normal"
            if normal_dir.exists():
                image_paths = sorted(normal_dir.glob("*.png")) + \
                              sorted(normal_dir.glob("*.jpg"))
                labels = [0] * len(image_paths)
        else:
            # Normal + defect images for supervised training
            for label_idx, class_name in enumerate(["normal", "defect"]):
                class_dir = split_dir / class_name
                if class_dir.exists():
                    paths = sorted(class_dir.glob("*.png")) + \
                            sorted(class_dir.glob("*.jpg"))
                    image_paths.extend(paths)
                    labels.extend([label_idx] * len(paths))

        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.mode == "ssl":
            return image  # SSL returns transformed views

        return image, self.labels[idx]


class MVTecDataset(IndustrialDataset):
    """
    MVTec AD dataset loader.
    Structure: mvtec/<category>/train/good/, mvtec/<category>/test/<defect_type>/
    """

    CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]

    def __init__(self, root_dir: str, category: str, **kwargs):
        assert category in self.CATEGORIES, f"Unknown category: {category}"
        self.category = category
        super().__init__(root_dir=os.path.join(root_dir, category), **kwargs)

    def _load_file_list(self):
        image_paths, labels = [], []

        if self.split == "train":
            good_dir = self.root_dir / "train" / "good"
            if good_dir.exists():
                paths = sorted(good_dir.glob("*.png"))
                image_paths.extend(paths)
                labels.extend([0] * len(paths))
        else:
            test_dir = self.root_dir / "test"
            if test_dir.exists():
                for defect_dir in sorted(test_dir.iterdir()):
                    label = 0 if defect_dir.name == "good" else 1
                    paths = sorted(defect_dir.glob("*.png"))
                    image_paths.extend(paths)
                    labels.extend([label] * len(paths))

        return image_paths, labels


class SSLTransform:
    """
    Returns two augmented views of the same image for contrastive SSL.
    """

    def __init__(self, image_size: int = 224, strength: float = 0.5):
        color_jitter = T.ColorJitter(
            brightness=0.8 * strength,
            contrast=0.8 * strength,
            saturation=0.8 * strength,
            hue=0.2 * strength,
        )
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=int(0.1 * image_size) | 1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def get_eval_transform(image_size: int = 224) -> T.Compose:
    """Standard evaluation transform (no augmentation)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
