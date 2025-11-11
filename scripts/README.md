# Scripts Directory

Command-line tools for training, inference, and evaluation of VisDrone object detection models.

## Available Scripts

### 1. `train.py` - Train Detection Models

Train object detection models on VisDrone dataset with support for multiple architectures.

**Basic Usage:**

```bash
python scripts/train.py \
    --train-img-dir data/VisDrone2019-DET-train/images \
    --train-ann-dir data/VisDrone2019-DET-train/annotations \
    --val-img-dir data/VisDrone2019-DET-val/images \
    --val-ann-dir data/VisDrone2019-DET-val/annotations \
    --model fasterrcnn_resnet50 \
    --epochs 50 \
    --batch-size 4 \
    --output-dir outputs/fasterrcnn
```

**Supported Models:**

- `fasterrcnn_resnet50` - Faster R-CNN with ResNet50-FPN backbone
- `fasterrcnn_mobilenet` - Faster R-CNN with MobileNetV3-Large-FPN (faster, lighter)
- `fcos_resnet50` - FCOS (anchor-free) with ResNet50-FPN
- `retinanet_resnet50` - RetinaNet with ResNet50-FPN

**Key Options:**

- `--amp` - Use automatic mixed precision (faster training on modern GPUs)
- `--pretrained` - Use COCO pretrained weights (recommended)
- `--resume checkpoint.pth` - Resume training from checkpoint
- `--lr 0.005` - Learning rate
- `--save-every 5` - Save checkpoint every N epochs

---

### 2. `inference.py` - Run Inference

Run trained models on images or videos.

**Image Inference:**

```bash
# Single image
python scripts/inference.py \
    --checkpoint outputs/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input test_image.jpg \
    --output-dir results

# Directory of images
python scripts/inference.py \
    --checkpoint outputs/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input data/test_images/ \
    --output-dir results
```

**Video Inference:**

```bash
python scripts/inference.py \
    --checkpoint outputs/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input video.mp4 \
    --output-dir results
```

**Key Options:**

- `--score-threshold 0.5` - Confidence threshold for detections
- `--show` - Display results in window
- `--no-save-viz` - Don't save visualization images

---

### 3. `webcam_demo.py` - Real-time Webcam Demo

Real-time object detection using your webcam. Perfect for testing if everything works!

**Basic Usage:**

```bash
# With trained model
python scripts/webcam_demo.py \
    --checkpoint outputs/best_model.pth \
    --model fasterrcnn_resnet50

# With pretrained COCO weights (no training needed!)
python scripts/webcam_demo.py \
    --model fasterrcnn_resnet50
```

**Controls:**

- `q` - Quit
- `s` - Save current frame
- `SPACE` - Pause/Resume

**Key Options:**

- `--camera 0` - Camera index (default: 0)
- `--width 640 --height 480` - Resolution
- `--score-threshold 0.5` - Detection threshold
- `--no-display-fps` - Hide FPS counter

---

### 4. `convert_annotations.py` - Convert Annotations

Convert VisDrone annotations to COCO or YOLO format.

**COCO Format:**

```bash
python scripts/convert_annotations.py \
    --format coco \
    --image-dir data/images \
    --annotation-dir data/annotations \
    --output annotations_coco.json \
    --validate
```

**YOLO Format:**

```bash
python scripts/convert_annotations.py \
    --format yolo \
    --image-dir data/images \
    --annotation-dir data/annotations \
    --output-dir data/yolo_labels \
    --validate
```

**Key Options:**

- `--keep-ignored` - Keep boxes marked as ignored (score=0)
- `--keep-crowd` - Keep crowd/ignored regions (category=0)
- `--validate` - Validate output after conversion

---

### 5. `evaluate.py` - Evaluate Models

Compute detection metrics on validation/test sets.

**Basic Usage:**

```bash
python scripts/evaluate.py \
    --checkpoint outputs/best_model.pth \
    --model fasterrcnn_resnet50 \
    --image-dir data/VisDrone2019-DET-val/images \
    --annotation-dir data/VisDrone2019-DET-val/annotations \
    --output-dir eval_results
```

**Metrics Computed:**

- Precision, Recall, F1-Score (overall and per-class)
- True Positives, False Positives, False Negatives
- Inference time and FPS

**Key Options:**

- `--score-threshold 0.05` - Score threshold for detections
- `--iou-threshold 0.5` - IoU threshold for matching
- `--save-predictions` - Save predictions to JSON
- `--batch-size 4` - Batch size for evaluation

---

## Quick Start Example

**1. Download VisDrone dataset:**

```bash
# Download from: https://github.com/VisDrone/VisDrone-Dataset
# Extract to data/ directory
```

**2. Test webcam (no training needed):**

```bash
python scripts/webcam_demo.py --model fasterrcnn_mobilenet
```

**3. Train a model:**

```bash
python scripts/train.py \
    --train-img-dir data/VisDrone2019-DET-train/images \
    --train-ann-dir data/VisDrone2019-DET-train/annotations \
    --val-img-dir data/VisDrone2019-DET-val/images \
    --val-ann-dir data/VisDrone2019-DET-val/annotations \
    --model fasterrcnn_mobilenet \
    --epochs 30 \
    --batch-size 8 \
    --amp \
    --output-dir outputs/mobilenet
```

**4. Evaluate:**

```bash
python scripts/evaluate.py \
    --checkpoint outputs/mobilenet/best_model.pth \
    --model fasterrcnn_mobilenet \
    --image-dir data/VisDrone2019-DET-val/images \
    --annotation-dir data/VisDrone2019-DET-val/annotations
```

**5. Run inference:**

```bash
python scripts/inference.py \
    --checkpoint outputs/mobilenet/best_model.pth \
    --model fasterrcnn_mobilenet \
    --input test_images/ \
    --output-dir results
```

---

## Tips

### GPU Memory Issues?

- Reduce `--batch-size` (try 2 or 1)
- Use `fasterrcnn_mobilenet` instead of `fasterrcnn_resnet50`
- Reduce image resolution in dataset preprocessing

### Want Faster Training?

- Use `--amp` for automatic mixed precision
- Use `fasterrcnn_mobilenet` for lighter model
- Increase `--batch-size` if GPU memory allows

### Want Better Accuracy?

- Train for more epochs (50-100)
- Use `fasterrcnn_resnet50` or `retinanet_resnet50`
- Use data augmentation (implement custom transforms)
- Fine-tune learning rate and schedule

### Real-time Inference?

- Use `fasterrcnn_mobilenet` (fastest)
- Lower `--score-threshold` to reduce post-processing
- Reduce input resolution
