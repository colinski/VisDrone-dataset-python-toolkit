"""Real-time webcam demo for VisDrone object detection.

Press 'q' to quit, 's' to save a frame.
"""

from __future__ import annotations

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import torch

from visdrone_toolkit.utils import VISDRONE_CLASSES, get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time webcam detection demo")

    # Model
    parser.add_argument("--checkpoint", help="Path to model checkpoint (optional)")
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

    # Webcam
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")

    # Inference
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)"
    )

    # Display
    parser.add_argument("--no-display-fps", action="store_true", help="Don't display FPS counter")
    parser.add_argument(
        "--save-dir", default="webcam_captures", help="Directory to save captured frames"
    )

    return parser.parse_args()


class FPSCounter:
    """Simple FPS counter using a sliding window."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: deque = deque(maxlen=window_size)
        self.last_time = time.time()

    def update(self):
        """Update FPS counter."""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

    def get_fps(self) -> float:
        """Get current FPS."""
        if len(self.frame_times) == 0:
            return 0.0
        return float(len(self.frame_times) / sum(self.frame_times))


def load_model(checkpoint_path: str, model_name: str, num_classes: int, device: torch.device):
    """Load model from checkpoint or create pretrained model."""
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}...")
        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print("✓ Model loaded from checkpoint")
    else:
        print("Creating pretrained model (COCO weights)...")
        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
        )
        print("✓ Pretrained model loaded")
        print("Note: Using COCO pretrained weights. Train on VisDrone for better results!")

    model.to(device)
    model.eval()
    return model


def draw_detections(frame, boxes, labels, scores, score_threshold: float = 0.5):
    """Draw bounding boxes and labels on frame."""
    h, w = frame.shape[:2]

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)

        # Clip to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Get class name
        class_name = VISDRONE_CLASSES[label] if label < len(VISDRONE_CLASSES) else f"class_{label}"

        # Choose color based on class
        color = (0, 255, 0)  # Default green
        if label == 1 or label == 2:  # pedestrian, people
            color = (0, 0, 255)  # Red
        elif label >= 4 and label <= 10:  # vehicles
            color = (255, 0, 0)  # Blue

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_text = f"{class_name}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Ensure label is within frame
        label_y1 = max(y1 - text_height - 4, 0)
        label_y2 = label_y1 + text_height + 4

        cv2.rectangle(frame, (x1, label_y1), (x1 + text_width, label_y2), color, -1)
        cv2.putText(
            frame,
            label_text,
            (x1, label_y2 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model = load_model(args.checkpoint, args.model, args.num_classes, device)

    # Open webcam
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera opened: {actual_width}x{actual_height}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # FPS counter
    fps_counter = FPSCounter()

    # Display instructions
    print(f"\n{'=' * 60}")
    print("Webcam Demo Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  ' ' - Pause/Resume")
    print(f"{'=' * 60}\n")

    paused = False
    frame_count = 0
    saved_count = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                frame_count += 1

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to tensor
                image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.to(device)

                # Run inference
                with torch.no_grad():
                    predictions = model([image_tensor])[0]

                # Get predictions
                boxes = predictions["boxes"].cpu().numpy()
                labels = predictions["labels"].cpu().numpy()
                scores = predictions["scores"].cpu().numpy()

                # Filter by score
                mask = scores >= args.score_threshold
                boxes = boxes[mask]
                labels = labels[mask]
                scores = scores[mask]

                # Draw detections
                frame = draw_detections(frame, boxes, labels, scores, args.score_threshold)

                # Update FPS
                fps_counter.update()
                current_fps = fps_counter.get_fps()

                # Draw FPS and detection count
                if not args.no_display_fps:
                    info_text = f"FPS: {current_fps:.1f} | Detections: {len(boxes)}"
                    cv2.putText(
                        frame,
                        info_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # Display frame
            cv2.imshow("VisDrone Webcam Demo", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("s"):
                # Save frame
                saved_count += 1
                save_path = save_dir / f"capture_{saved_count:04d}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f"✓ Frame saved to {save_path}")
            elif key == ord(" "):
                # Toggle pause
                paused = not paused
                if paused:
                    print("⏸ Paused")
                else:
                    print("▶ Resumed")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        print(f"\n{'=' * 60}")
        print("Session Summary:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Frames saved: {saved_count}")
        if frame_count > 0:
            print(f"  Average FPS: {fps_counter.get_fps():.2f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
