from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def resolve_image_dir(data_dir: Union[str, Path], folder_name: str) -> Path:
    """
    Return /data/final/train_images or /data/final/test_images.

    Important:
    Do NOT automatically return train_images/train_images or test_images/test_images.
    KaggleImageDataset searches recursively, so returning the top folder is safer.
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / folder_name

    if image_dir.is_dir():
        return image_dir

    raise FileNotFoundError(f"Image directory not found: {image_dir}")


def list_image_files(image_dir: Union[str, Path]) -> list[Path]:
    image_dir = Path(image_dir)
    return sorted(
        [
            p
            for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name,
    )


def build_image_index(image_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Map image filename to path.
    If duplicate filenames exist, prefer the shallower path.
    """
    index: Dict[str, Path] = {}

    for p in list_image_files(image_dir):
        key = p.name

        if key not in index:
            index[key] = p
        else:
            old = index[key]
            if len(p.parts) < len(old.parts):
                index[key] = p

    return index


def find_image_path(
    image_dir: Union[str, Path],
    image_id: Union[str, int],
    image_index: Optional[Dict[str, Path]] = None,
) -> Path:
    image_dir = Path(image_dir)
    image_id = str(image_id)

    if image_index is not None and image_id in image_index:
        return image_index[image_id]

    path = image_dir / image_id
    if path.is_file():
        return path

    stem = Path(image_id).stem

    if image_index is not None:
        for ext in IMAGE_EXTENSIONS:
            candidate_name = f"{stem}{ext}"
            if candidate_name in image_index:
                return image_index[candidate_name]

    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate

    matches = sorted(
        [
            p
            for p in image_dir.rglob(f"{stem}.*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: (len(p.parts), str(p)),
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(f"Image file not found for id '{image_id}' in {image_dir}")


class KaggleImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Union[str, Path],
        id_col: str = "id",
        label_col: Optional[str] = None,
        transform=None,
        labels: Optional[Sequence[int]] = None,
    ):
        if id_col not in dataframe.columns:
            raise ValueError(f"Dataframe must contain '{id_col}' column")

        if label_col is not None and label_col not in dataframe.columns and labels is None:
            raise ValueError(f"Dataframe must contain '{label_col}' column")

        self.image_dir = Path(image_dir)
        self.ids = dataframe[id_col].astype(str).tolist()
        self.transform = transform

        self.image_index = build_image_index(self.image_dir)
        self.paths = [
            find_image_path(self.image_dir, image_id, self.image_index)
            for image_id in self.ids
        ]

        if labels is not None:
            self.labels = [int(label) for label in labels]
        elif label_col is not None:
            self.labels = dataframe[label_col].astype(int).tolist()
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.labels is None:
            return image, self.ids[index]

        return image, self.labels[index]