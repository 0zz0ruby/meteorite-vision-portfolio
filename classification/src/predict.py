import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny
from tqdm import tqdm

from dataset import KaggleImageDataset, build_image_index, list_image_files, resolve_image_dir


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict with a trained DINOv2 or ConvNeXt-Tiny checkpoint."
    )
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="submission.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_image_folder", type=str, default="test_images",
                        help="Subfolder under data_dir containing test images "
                             "(e.g. test_images_bg_removed).")
    parser.add_argument("--test_bg", type=str, default="black",
                        choices=["black", "white"],
                        help="Background color of test images. 'white' will be converted to black.")
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device, but CUDA is not available.")

    return device


def load_checkpoint(path: Path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


class DINOv2Classifier(nn.Module):
    """Frozen DINOv2 backbone + trainable Linear head (supports multi-layer CLS concat)."""

    def __init__(self, backbone: nn.Module, in_features: int, num_classes: int,
                 concat_layers: int = 2):
        super().__init__()
        self.backbone = backbone
        self.concat_layers = concat_layers
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone.prepare_tokens_with_masks(x)
        cls_tokens = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i >= len(self.backbone.blocks) - self.concat_layers:
                cls_tokens.append(x[:, 0, :])
        features = torch.cat(cls_tokens, dim=-1)
        return self.head(features)


def build_model(model_name: str, num_classes: int, concat_layers: int = 2) -> nn.Module:
    if model_name.startswith("dinov2"):
        base_dim = DINOV2_DIMS[model_name]
        in_features = base_dim * concat_layers
        backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        backbone.eval()
        return DINOv2Classifier(backbone, in_features, num_classes,
                                concat_layers=concat_layers)

    if model_name == "convnext_tiny":
        model = convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")


def _white_bg_to_black(pil_img: Image.Image) -> Image.Image:
    """Replace near-white background pixels (>240 in all channels) with black."""
    arr = np.array(pil_img)
    mask = (arr > 240).all(axis=2)
    arr[mask] = [0, 0, 0]
    return Image.fromarray(arr)


def get_transform(image_size: int, test_bg: str = "black"):
    ops = []
    if test_bg == "white":
        ops.append(transforms.Lambda(_white_bg_to_black))
    ops += [
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def normalize_output_path(output_path) -> Path:
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".csv":
        output_path = output_path / "submission.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def log_message(logger, message, *args):
    if logger is not None:
        logger.info(message, *args)
    else:
        print(message % args if args else message)


def make_test_df(data_dir: Path, test_image_dir: Path, logger=None) -> pd.DataFrame:
    """Return test IDs strictly from sample_submission.csv (Kaggle requirement)."""
    sample_path = data_dir / "sample_submission.csv"

    if sample_path.is_file():
        sample_df = pd.read_csv(sample_path)
        if "id" not in sample_df.columns:
            raise ValueError("sample_submission.csv must contain 'id' column")

        ids = sample_df["id"].astype(str).tolist()
        log_message(logger, "sample_submission row count: %d", len(ids))
        return pd.DataFrame({"id": ids})

    # Fallback: no sample_submission.csv — use real images
    real_test_ids = [p.name for p in list_image_files(test_image_dir)]
    if not real_test_ids:
        raise FileNotFoundError(f"No test images found in {test_image_dir}")
    log_message(logger, "No sample_submission.csv; using real test images: %d", len(real_test_ids))
    return pd.DataFrame({"id": sorted(real_test_ids)})


def make_tta_batches(images: torch.Tensor):
    """
    images shape: [B, C, H, W]
    TTA variants:
    1. original
    2. horizontal flip
    3. vertical flip
    4. horizontal + vertical flip
    """
    return [
        images,
        torch.flip(images, dims=[3]),
        torch.flip(images, dims=[2]),
        torch.flip(images, dims=[2, 3]),
    ]


@torch.no_grad()
def predict(model, loader, classes, device, use_amp, tta=True):
    model.eval()
    id_to_label = {}

    for images, image_ids in tqdm(loader, desc="predict", leave=False):
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if tta:
                prob_sum = None
                for tta_images in make_tta_batches(images):
                    outputs = model(tta_images)
                    probs = torch.softmax(outputs, dim=1)
                    prob_sum = probs if prob_sum is None else prob_sum + probs
                avg_probs = prob_sum / 4.0
                preds = avg_probs.argmax(dim=1).cpu().numpy().tolist()
            else:
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy().tolist()

        for image_id, pred in zip(image_ids, preds):
            label = classes[pred] if classes is not None else int(pred)
            id_to_label[str(image_id)] = label

    return id_to_label


def run_prediction(
    data_dir,
    checkpoint,
    batch_size=64,
    image_size=None,
    output_path="submission.csv",
    device_arg="auto",
    num_workers=2,
    logger=None,
    tta=True,
    test_image_folder="test_images",
    test_bg="black",
):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint)
    output_path = normalize_output_path(output_path)

    device = resolve_device(device_arg)
    use_amp = device.type == "cuda"

    test_image_dir = resolve_image_dir(data_dir, test_image_folder)
    test_df = make_test_df(data_dir, test_image_dir, logger=logger)

    checkpoint_data = load_checkpoint(checkpoint_path, device)

    num_classes = int(checkpoint_data["num_classes"])
    model_name = checkpoint_data.get("model_name", "convnext_tiny")
    classes = checkpoint_data.get("classes")
    concat_layers = checkpoint_data.get("concat_layers", 2)
    image_size = image_size or int(checkpoint_data.get("image_size", 224))

    log_message(logger, "Prediction data directory: %s", data_dir)
    log_message(logger, "Test image directory: %s", test_image_dir)
    log_message(logger, "Checkpoint: %s", checkpoint_path)
    log_message(logger, "Output path: %s", output_path)
    log_message(logger, "Device: %s | AMP: %s", device, use_amp)
    log_message(logger, "TTA enabled: %s", tta)

    model = build_model(model_name, num_classes, concat_layers=concat_layers)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)

    # Separate ids that have real images vs missing ones
    image_index = build_image_index(test_image_dir)
    has_image = test_df["id"].astype(str).isin(image_index)
    df_predict = test_df[has_image].copy()
    df_missing = test_df[~has_image].copy()

    if len(df_missing) > 0:
        log_message(logger, "WARNING: %d ids in sample_submission.csv have no image. "
                    "They will be assigned the default class.", len(df_missing))

    test_dataset = KaggleImageDataset(
        df_predict,
        test_image_dir,
        transform=get_transform(image_size, test_bg=test_bg),
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    test_loader = DataLoader(test_dataset, **loader_kwargs)

    id_to_label = predict(
        model,
        test_loader,
        classes,
        device,
        use_amp,
        tta=tta,
    )

    submission = test_df[["id"]].copy()
    submission["label"] = submission["id"].astype(str).map(id_to_label)

    # Fill missing-image ids with default class (first class)
    default_label = classes[0] if classes is not None else 0
    na_mask = submission["label"].isna()
    if na_mask.any():
        log_message(logger, "Filling %d missing predictions with default class '%s'.",
                    na_mask.sum(), default_label)
        submission.loc[na_mask, "label"] = default_label

    submission.to_csv(output_path, index=False)

    log_message(logger, "Saved submission: %s", output_path)
    log_message(logger, "Saved rows: %d", len(submission))

    return output_path


def main():
    args = parse_args()

    run_prediction(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        image_size=args.image_size,
        output_path=args.output_path,
        device_arg=args.device,
        num_workers=args.num_workers,
        tta=not args.no_tta,
        test_image_folder=args.test_image_folder,
        test_bg=args.test_bg,
    )


if __name__ == "__main__":
    main()
