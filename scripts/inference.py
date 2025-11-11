"""
Inference script for VisDrone object detection models.

Supports inference on:
- Single images
- Multiple images in a directory
- Video files
"""
import argparse
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

from visdrone_toolkit.utils import VISDRONE_CLASSES, get_model
from visdrone_toolkit.visualization import visualize_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on VisDrone models")

    # Model
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--model",
        default="fasterrcnn_resnet50",
        choices=[
            "fasterrcnn_resnet50",
            "fasterrcnn_mobilenet",
            "fcos_resnet50",
            "retinanet_resnet50",
        ],
        help="Model architecture",
    )
    parser.add_argument("--num-classes", type=int, default=12, help="Number of classes")

    # Input
    parser.add_argument("--input", required=True, help="Input image/directory/video")
    parser.add_argument("--output-dir", default="inference_outputs", help="Output directory")

    # Inference parameters
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)"
    )

    # Visualization
    parser.add_argument("--no-save-viz", action="store_true", help="Don't save visualizations")
    parser.add_argument("--show", action="store_true", help="Display results")

    return parser.parse_args()


def load_model(checkpoint_path: str, model_name: str, num_classes: int, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Create model
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    return model


@torch.no_grad()
def run_inference_on_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    score_threshold: float = 0.5,
) -> dict:
    """Run inference on a single image."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Convert to tensor
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(device)

    # Run inference
    start_time = time.time()
    predictions = model([image_tensor])[0]
    inference_time = time.time() - start_time

    # Filter by score
    mask = predictions["scores"] >= score_threshold
    predictions = {
        "boxes": predictions["boxes"][mask],
        "labels": predictions["labels"][mask],
        "scores": predictions["scores"][mask],
    }

    return {
        "predictions": predictions,
        "image": image_np,
        "inference_time": inference_time,
    }


def process_images(
    model: torch.nn.Module,
    input_path: Union[str, Path],
    output_dir: Path,
    device: torch.device,
    score_threshold: float,
    save_viz: bool,
    show: bool,
):
    """Process images from file or directory."""
    input_path = Path(input_path)

    # Get image files
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(
            [
                f
                for f in input_path.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ]
        )
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    if len(image_files) == 0:
        print("No images found!")
        return

    print(f"\nProcessing {len(image_files)} images...")
    print(f"{'='*60}")

    total_inference_time = 0
    total_detections = 0

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {image_path.name}")

        # Run inference
        result = run_inference_on_image(model, image_path, device, score_threshold)

        num_detections = len(result["predictions"]["boxes"])
        total_detections += num_detections
        total_inference_time += result["inference_time"]

        print(f"  Detections: {num_detections}")
        print(f"  Inference time: {result['inference_time']*1000:.2f}ms")

        # Visualize and save
        if save_viz:
            output_path = output_dir / f"{image_path.stem}_result.jpg"
            visualize_predictions(
                result["image"],
                result["predictions"]["boxes"],
                result["predictions"]["labels"],
                result["predictions"]["scores"],
                score_threshold=score_threshold,
                save_path=output_path,
                show=show,
            )
            print(f"  ✓ Saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average inference time: {(total_inference_time/len(image_files))*1000:.2f}ms")
    print(f"  FPS: {len(image_files)/total_inference_time:.2f}")


def process_video(
    model: torch.nn.Module,
    video_path: str | Path,
    output_dir: Path,
    device: torch.device,
    score_threshold: float,
):
    """Process video file."""
    video_path = Path(video_path)
    output_path = Path(output_dir) / f"{video_path.stem}_result.mp4"

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    total_inference_time = 0.0

    print(f"\n{'='*60}")
    print("Processing frames...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to tensor
            image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.to(device)

            # Run inference
            start_time = time.time()
            predictions = model([image_tensor])[0]
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # Filter by score
            mask = predictions["scores"] >= score_threshold
            boxes = predictions["boxes"][mask].cpu().numpy()
            labels = predictions["labels"][mask].cpu().numpy()
            scores = predictions["scores"][mask].cpu().numpy()

            # Draw detections
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.astype(int)

                # Get class name and color
                class_name = (
                    VISDRONE_CLASSES[label] if label < len(VISDRONE_CLASSES) else f"class_{label}"
                )
                color = (0, 255, 0)  # Green

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label_text = f"{class_name}: {score:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            # Write frame
            out.write(frame)

            # Print progress
            if frame_count % 30 == 0 or frame_count == total_frames:
                avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
                print(
                    f"  Frame {frame_count}/{total_frames} - "
                    f"Avg FPS: {avg_fps:.2f} - "
                    f"Detections: {len(boxes)}"
                )

    finally:
        cap.release()
        out.release()

    print(f"\n{'='*60}")
    print(f"✓ Video saved to {output_path}")
    print(f"  Processed {frame_count} frames")
    print(f"  Average inference FPS: {frame_count/total_inference_time:.2f}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, args.model, args.num_classes, device)

    # Check input type
    input_path = Path(args.input)

    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    # Process based on input type
    if input_path.is_file():
        if input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            # Video file
            process_video(model, input_path, output_dir, device, args.score_threshold)
        else:
            # Single image
            process_images(
                model,
                input_path,
                output_dir,
                device,
                args.score_threshold,
                not args.no_save_viz,
                args.show,
            )
    elif input_path.is_dir():
        # Directory of images
        process_images(
            model,
            input_path,
            output_dir,
            device,
            args.score_threshold,
            not args.no_save_viz,
            args.show,
        )
    else:
        raise ValueError(f"Invalid input: {input_path}")

    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
