"""
Training Script for Model 1: Cross-Encoder (Deep Ranking)
Multi-head transformer for resume-job scoring and classification.
Customized for user datasets with optimized data loading.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.cross_encoder import MultiHeadCrossEncoder, CrossEncoderTrainer
from data.loaders import CrossEncoderDataset, create_cross_encoder_loaders
from scripts.mlflow_tracking import (
    setup_experiment,
    log_params_from_config,
    log_training_metrics,
    log_artifacts,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Cross-Encoder Model')
    
    # Data paths - use config or command line overrides
    data_config = config.get('data', {}).get('raw', {}).get('cross_encoder', {})
    
    parser.add_argument(
        '--train_path',
        type=str,
        default=data_config.get('train', ''),
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--val_path',
        type=str,
        default=data_config.get('validation', ''),
        help='Path to validation data CSV (optional)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=config['models']['cross_encoder']['pretrained_model'],
        help='Pretrained model name'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=config['models']['cross_encoder']['epochs'],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config['models']['cross_encoder']['batch_size'],
        help='Training batch size'
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=config['models']['cross_encoder'].get('eval_batch_size', 32),
        help='Evaluation batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=config['models']['cross_encoder']['learning_rate'],
        help='Learning rate'
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=config['models']['cross_encoder']['max_seq_length'],
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=config['inference']['device'],
        help='Device to train on (cuda/cpu/auto)'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=config['models']['cross_encoder'].get('fp16', False),
        dest='fp16',
        help='Enable mixed precision training (fp16)'
    )
    parser.add_argument(
        '--no-fp16',
        action='store_false',
        dest='fp16',
        help='Disable mixed precision training'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/cross_encoder',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=5,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--val_every',
        type=int,
        default=1,
        help='Validate every N epochs'
    )
    parser.add_argument(
        '--regression_weight',
        type=float,
        default=0.5,
        help='Weight for regression loss'
    )
    parser.add_argument(
        '--classification_weight',
        type=float,
        default=0.5,
        help='Weight for classification loss'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=config['models']['cross_encoder'].get('gradient_accumulation_steps', 4),
        help='Number of gradient accumulation steps (reduces GPU memory usage when lowered)'
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        default=config['models']['cross_encoder'].get('torch_compile', False),
        dest='compile',
        help='Enable torch.compile (PyTorch 2.0+) for kernel fusion speedup'
    )
    parser.add_argument(
        '--no-compile',
        action='store_false',
        dest='compile',
        help='Disable torch.compile'
    )
    parser.add_argument(
        '--fused_adam',
        action='store_true',
        default=config['models']['cross_encoder'].get('fused_adam', False),
        dest='fused_adam',
        help='Use fused AdamW kernels (PyTorch 2.0+ CUDA)'
    )
    parser.add_argument(
        '--no-fused-adam',
        action='store_false',
        dest='fused_adam',
        help='Disable fused AdamW'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,  # raised from 0-on-Windows: pre-tokenized data means no tokenizer in __getitem__
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--balance',
        action='store_true',
        default=False,
        dest='balance',
        help='Oversample minority classes (Good Fit) and use class-weighted loss'
    )
    parser.add_argument(
        '--no-balance',
        action='store_false',
        dest='balance',
        help='Disable class balancing'
    )

    return parser.parse_args()


def setup_device(args):
    """Setup training device with proper configuration."""
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Configure for GPU if available
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    return device


def train(args):
    """Main training function."""
    logger.info("="*60)
    logger.info("Cross-Encoder Training Script")
    logger.info("="*60)
    
    # Validate data paths
    if not args.train_path or not os.path.exists(args.train_path):
        logger.error(f"Training data not found: {args.train_path}")
        sys.exit(1)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = setup_device(args)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log data paths
    logger.info(f"Training data: {args.train_path}")
    logger.info(f"Validation data: {args.val_path if args.val_path else 'Using split from training data'}")

    # ── MLflow setup ─────────────────────────────────────────────────────
    import mlflow
    exp_prefix = config.get("mlflow", {}).get("experiment_name_prefix", "")
    experiment_name = f"{exp_prefix}cross-encoder"
    setup_experiment(config, experiment_name)
    mlflow_run = mlflow.start_run(run_name=f"run_{datetime.now():%Y%m%d_%H%M%S}")
    run_id = mlflow_run.info.run_id
    logger.info("MLflow run %s started (experiment: %s)", run_id[:8], experiment_name)

    # Merge config defaults with CLI overrides (CLI takes precedence)
    training_params = dict(config.get("models", {}).get("cross_encoder", {}))
    training_params.update({
        "train_path": args.train_path,
        "val_path": args.val_path,
        "epochs": str(args.epochs),
        "batch_size": str(args.batch_size),
        "learning_rate": str(args.learning_rate),
        "max_seq_length": str(args.max_seq_length),
        "gradient_accumulation_steps": str(args.gradient_accumulation_steps),
        "regression_weight": str(args.regression_weight),
        "classification_weight": str(args.classification_weight),
        "fp16": str(args.fp16),
        "torch_compile": str(args.compile),
        "fused_adam": str(args.fused_adam),
        "device": str(device),
    })
    mlflow.log_params(training_params)

    # Create data loaders using the new optimized loader
    logger.info("Creating data loaders...")
    loaders = create_cross_encoder_loaders(
        train_path=args.train_path,
        val_path=args.val_path if args.val_path else None,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_seq_length=args.max_seq_length,
        tokenizer_name=args.model_name,
        config=config,
        balance=args.balance
    )
    
    if 'train' not in loaders:
        logger.error("Failed to create training data loader")
        sys.exit(1)
    
    train_loader = loaders['train']
    val_loader = loaders.get('val')
    
    logger.info(f"Training batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")

    # ── Class weights for imbalanced training ────────────────────────────
    class_weights = None
    if args.balance and 'label_counts' in loaders:
        counts = loaders['label_counts']
        total = sum(counts.values())
        n_classes = len(counts)
        # Inverse frequency: w_i = total / (n_classes * count_i)
        class_weights = torch.tensor(
            [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
            dtype=torch.float32,
        )
        logger.info(f"Class weights (inverse frequency): "
                    f"{dict(zip(['Bad Fit','Potential Fit','Good Fit'], class_weights.tolist()))}")

    # Initialize model
    logger.info(f"Initializing Cross-Encoder model: {args.model_name}")
    model = MultiHeadCrossEncoder(
        pretrained_model_name=args.model_name,
        num_classes=3,
        hidden_size=config['models']['cross_encoder'].get('hidden_size', 768),
        dropout=config['models']['cross_encoder'].get('dropout', 0.1)
    )
    model.to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    mlflow.log_params({"total_parameters": str(total_params), "trainable_parameters": str(trainable_params),})
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    total_steps = (len(train_loader) * args.epochs) // args.gradient_accumulation_steps
    trainer = CrossEncoderTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=config['models']['cross_encoder'].get('weight_decay', 0.01),
        warmup_steps=config['models']['cross_encoder'].get('warmup_steps', 500),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        device=device,
        total_training_steps=total_steps,
        fused_adam=args.fused_adam,
        torch_compile=args.compile,
        class_weights=class_weights,
    )
    
    # Training loop — best model tracked by val loss with early stopping
    ce_cfg = config['models']['cross_encoder']
    best_val_loss = float('inf')
    best_metrics = {}
    training_history = []
    early_stopping_patience = ce_cfg.get('early_stopping_patience', 0)  # 0 = disabled
    epochs_no_improve = 0
    epochs_completed = 0
    early_stop_triggered = False

    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60)

    for epoch in range(1, args.epochs + 1):
        epochs_completed = epoch
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 40)

        # Train
        train_metrics = trainer.train_epoch(
            train_loader,
            epoch=epoch,
            regression_weight=args.regression_weight,
            classification_weight=args.classification_weight
        )

        logger.info(f"Train - Loss: {train_metrics['avg_loss']:.4f}, "
                   f"Reg Loss: {train_metrics.get('regression_loss', 0):.4f}, "
                   f"Cls Loss: {train_metrics.get('classification_loss', 0):.4f}")

        # Validate
        val_metrics = None
        if val_loader and epoch % args.val_every == 0:
            logger.info("Starting validation (%d batches, batch_size=%d)...",
                        len(val_loader), val_loader.batch_size)
            val_metrics = trainer.evaluate(val_loader)
            logger.info("Validation complete.")
            logger.info(f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
                       f"MSE: {val_metrics.get('mse', 0):.4f}, "
                       f"MAE: {val_metrics.get('mae', 0):.4f}, "
                       f"Accuracy: {val_metrics.get('accuracy', 0):.4f}, "
                       f"F1 (Macro): {val_metrics.get('f1_macro', 0):.4f}")

            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_metrics = val_metrics.copy()
                epochs_no_improve = 0

                # Save best checkpoint
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
                model.save(checkpoint_path)
                logger.info(f"  >> New best model saved! Loss: {best_val_loss:.4f}")
            elif early_stopping_patience > 0:
                epochs_no_improve += 1
                logger.info(
                    f"  No improvement for {epochs_no_improve}/{early_stopping_patience} epochs"
                )
                if epochs_no_improve >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    epochs_no_improve = 0  # reset for flag check
                    early_stop_triggered = True

        # Log epoch metrics to MLflow
        epoch_metrics = {
            "train_loss": train_metrics["avg_loss"],
            "train_regression_loss": train_metrics.get("regression_loss", 0),
            "train_classification_loss": train_metrics.get("classification_loss", 0),
        }
        if val_metrics:
            epoch_metrics.update({
                "val_loss": val_metrics["val_loss"],
                "val_mse": val_metrics.get("mse", 0),
                "val_accuracy": val_metrics.get("accuracy", 0),
                "val_f1_macro": val_metrics.get("f1_macro", 0),
                "val_f1_weighted": val_metrics.get("f1_weighted", 0),
            })
        mlflow.log_metrics(epoch_metrics, step=epoch)

        # Log epoch metrics locally
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_metrics['avg_loss'],
            'val_loss': val_metrics['val_loss'] if val_metrics else None,
            'val_mse': val_metrics.get('mse') if val_metrics else None,
            'val_accuracy': val_metrics.get('accuracy') if val_metrics else None
        }
        training_history.append(epoch_record)

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            metrics_for_checkpoint = val_metrics if val_metrics else train_metrics
            trainer.save_checkpoint(checkpoint_path, epoch, metrics_for_checkpoint)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        if early_stop_triggered:
            break

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    model.save(final_path)
    logger.info(f"\nSaved final model to {final_path}")

    # Save training summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'args': vars(args),
        'best_val_loss': best_val_loss,
        'best_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in best_metrics.items()},
        'epochs_completed': epochs_completed,
        'training_history': training_history,
        'model_config': {
            'pretrained_model': args.model_name,
            'num_classes': 3,
            'max_seq_length': args.max_seq_length
        }
    }
    
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Log MLflow artifacts, register model, and end run ─────────────────
    log_artifacts(args.output_dir, artifact_path="checkpoints")
    mlflow.log_artifact(summary_path, artifact_path="summary")
    mlflow.log_metric("best_val_loss", best_val_loss)
    if best_metrics:
        for k, v in best_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"best_{k}", float(v))

    # Model Registry: log best model with signature and auto-register
    mlflow_cfg = config.get("mlflow", {})
    if mlflow_cfg.get("register_models", False):
        from mlflow.models import infer_signature
        sample_input = {
            "input_ids": torch.zeros(1, args.max_seq_length, dtype=torch.long, device=device),
            "attention_mask": torch.ones(1, args.max_seq_length, dtype=torch.long, device=device),
        }
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input["input_ids"], sample_input["attention_mask"])
        sig = infer_signature(
            sample_input,
            {
                "regression_score": sample_output["regression_score"].cpu().numpy(),
                "classification_logits": sample_output["classification_logits"].cpu().numpy(),
            },
        )
        reg_name = f"{exp_prefix}cross-encoder"
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=sig,
            registered_model_name=reg_name,
        )
        logger.info("Model registered as '%s' in MLflow Model Registry", reg_name)

    mlflow.end_run()

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    if best_metrics:
        logger.info(f"Best Val Accuracy: {best_metrics.get('accuracy', 'N/A'):.4f}")
        logger.info(f"Best Val F1 (Macro): {best_metrics.get('f1_macro', 'N/A'):.4f}")
        logger.info(f"Best Val MSE: {best_metrics.get('mse', 'N/A'):.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info("="*60)
    
    return model


if __name__ == "__main__":
    args = parse_args()
    
    # Quick validation
    if not args.train_path:
        print("Error: --train_path is required")
        print("Example: python scripts/train_cross_encoder.py --train_path datasets/score dataset/train.csv")
        sys.exit(1)
    
    train(args)
