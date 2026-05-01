import argparse

import albumentations as A
from albumentations.pytorch import ToTensorV2

from visdrone_toolkit.dataset import VisDroneDataset


def build_transforms(image_size: int, train: bool) -> A.Compose:
    transforms = []
    if train:
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ]
    transforms += [
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="data/VisDrone2019-DET-val/images")
    p.add_argument("--annotation-dir", default="data/VisDrone2019-DET-val/annotations")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--train", action="store_true")
    args = p.parse_args()

    ds = VisDroneDataset(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        transforms=build_transforms(args.image_size, args.train),
        relabel_classes=True,
    )

    print(f"classes ({len(ds.classes)}): {ds.classes}")
    image, target = ds[0]
    print(
        f"image: shape={tuple(image.shape)} dtype={image.dtype} "
        f"min={float(image.min()):.3f} max={float(image.max()):.3f}"
    )
    print(f"boxes: {tuple(target['boxes'].shape)}")
    print(f"labels[:10]: {target['labels'][:10].tolist()}")


if __name__ == "__main__":
    main()
