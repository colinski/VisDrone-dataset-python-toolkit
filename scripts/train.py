"""
Training script for VisDrone object detection models.

Supports Faster R-CNN, FCOS, and RetinaNet with various backbones.
Includes automatic mixed precision, learning rate scheduling, and checkpointing.
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from visdrone_toolkit.dataset import VisDroneDataset
from visdrone_toolkit.utils import collate_fn, get_model, load_checkpoint, save_checkpoint
from visdrone_toolkit.visualization import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train object detection models on VisDrone")

    # Dataset paths
    parser.add_argument("--train-img-dir", required=True, help="Training images directory")
    parser.add_argument("--train-ann-dir", required=True, help="Training annotations directory")
    parser.add_argument("--val-img-dir", help="Validation images directory")
    parser.add_argument("--val-ann-dir", help="Validation annotations directory")

    # Model configuration
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
    parser.add_argument(
        "--pretrained", action="store_true", default=True, help="Use pretrained weights"
    )
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Training options
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument(
        "--filter-ignored", action="store_true", default=True, help="Filter ignored boxes"
    )
    parser.add_argument(
        "--filter-crowd", action="store_true", default=True, help="Filter crowd regions"
    )

    # Checkpointing
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")

    # Device
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)"
    )

    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
) -> float:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = len(data_loader)

    print(f"\n{'='*60}")
    print(f"Epoch {epoch} - Training")
    print(f"{'='*60}")

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass with optional AMP
        if use_amp and scaler is not None:
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        total_loss += losses.item()

        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"Batch [{batch_idx + 1}/{num_batches}] - "
                f"Loss: {losses.item():.4f} - "
                f"Avg Loss: {avg_loss:.4f} - "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches

    print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s - Avg Loss: {avg_loss:.4f}")

    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    """Evaluate model on validation set."""
    model.train()  # Keep in train mode for loss computation

    total_loss = 0
    num_batches = len(data_loader)

    print(f"\n{'='*60}")
    print(f"Epoch {epoch} - Validation")
    print(f"{'='*60}")

    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()

        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"Batch [{batch_idx + 1}/{num_batches}] - "
                f"Loss: {losses.item():.4f} - "
                f"Avg Loss: {avg_loss:.4f}"
            )

    avg_loss = total_loss / num_batches
    print(f"\nValidation completed - Avg Loss: {avg_loss:.4f}")

    return avg_loss


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = VisDroneDataset(
        image_dir=args.train_img_dir,
        annotation_dir=args.train_ann_dir,
        filter_ignored=args.filter_ignored,
        filter_crowd=args.filter_crowd,
    )

    val_dataset = None
    if args.val_img_dir and args.val_ann_dir:
        val_dataset = VisDroneDataset(
            image_dir=args.val_img_dir,
            annotation_dir=args.val_ann_dir,
            filter_ignored=args.filter_ignored,
            filter_crowd=args.filter_crowd,
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )

    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.1,
    )

    # AMP scaler
    scaler = GradScaler() if args.amp and device.type == "cuda" else None
    if args.amp:
        print("Using Automatic Mixed Precision (AMP)")

    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch = (
            load_checkpoint(
                args.resume,
                model,
                optimizer,
                lr_scheduler,
                device=str(device),
            )
            + 1
        )

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}")

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, scaler, args.amp
        )
        train_losses.append(train_loss)

        # Validate
        if val_loader:
            val_loss = evaluate(model, val_loader, device, epoch)
            val_losses.append(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "best_model.pth"
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_path,
                    lr_scheduler,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
                print(f"✓ New best model saved! Val Loss: {val_loss:.4f}")

        # Update learning rate
        lr_scheduler.step()

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                checkpoint_path,
                lr_scheduler,
                train_loss=train_loss,
                val_loss=val_losses[-1] if val_losses else None,
            )

    # Save final model
    final_path = output_dir / "final_model.pth"
    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        final_path,
        lr_scheduler,
        train_loss=train_losses[-1],
        val_loss=val_losses[-1] if val_losses else None,
    )
    print(f"\n✓ Final model saved to {final_path}")

    # Plot training curves
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(
        train_losses, val_losses if val_losses else None, save_path=curves_path, show=False
    )
    print(f"✓ Training curves saved to {curves_path}")

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
