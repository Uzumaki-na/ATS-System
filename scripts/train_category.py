"""
Training Script for Model 2: Category Encoder
Classifies resumes into 24 job categories.
Customized for user datasets with optimized data loading.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import faulthandler

os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TRANSFORMERS_NO_FLAX', '1')
os.environ.setdefault('PYTHONUNBUFFERED', '1')
os.environ.setdefault('TORCH_DISABLE_MKLDNN', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.category_encoder import CategoryEncoder, CategoryEncoderTrainer
from data.loaders import CategoryDataset, CombinedCategoryDataset, create_category_loaders
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

faulthandler.enable()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Category Encoder Model')
    
    # Data path from config
    data_config = config.get('data', {}).get('raw', {})
    
    parser.add_argument(
        '--csv_path',
        type=str,
        default=data_config.get('category_csv', ''),
        help='Path to category data CSV (optional)'
    )
    parser.add_argument(
        '--pdf_dir',
        type=str,
        default=data_config.get('category_pdf_dir', ''),
        help='Directory containing PDF resumes organized by category (optional)'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=data_config.get('category_csv', ''),
        help='Path to category data CSV (for backward compatibility)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=config['models']['category_encoder']['pretrained_model'],
        help='Pretrained model name'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=config['models']['category_encoder']['epochs'],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config['models']['category_encoder']['batch_size'],
        help='Training batch size'
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=config['models']['category_encoder'].get('eval_batch_size', 64),
        help='Evaluation batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=config['models']['category_encoder']['learning_rate'],
        help='Learning rate'
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=config['models']['category_encoder']['max_seq_length'],
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=config['inference']['device'],
        help='Device to train on (cuda/cpu/auto)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/category_encoder',
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
        '--train_split',
        type=float,
        default=0.8,
        help='Proportion of training data'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Proportion of validation data'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0 if os.name == 'nt' else 4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def setup_device(args):
    """Setup training device with proper configuration."""
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    return device


def train(args):
    """Main training function."""
    logger.info("="*60)
    logger.info("Category Encoder Training Script")
    logger.info("="*60)
    
    # Determine data sources (CSV and/or PDF)
    csv_path = args.csv_path or args.data_path  # Support backward compatibility
    pdf_dir = args.pdf_dir
    
    # Check if at least one data source is available
    has_csv = csv_path and os.path.exists(csv_path)
    has_pdf = pdf_dir and os.path.exists(pdf_dir)
    
    if not has_csv and not has_pdf:
        logger.error(f"No valid data source found.")
        logger.error(f"CSV path: {csv_path} (exists: {has_csv})")
        logger.error(f"PDF dir: {pdf_dir} (exists: {has_pdf})")
        sys.exit(1)
    
    # Log data sources
    logger.info(f"Data sources:")
    logger.info(f"  - CSV: {csv_path if has_csv else 'Not provided'}")
    logger.info(f"  - PDF: {pdf_dir if has_pdf else 'Not provided'}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = setup_device(args)
    logger.info(f"Using device: {device}")

    pin_memory = device.type == 'cuda'

    if device.type == 'cpu':
        try:
            torch.backends.mkldnn.enabled = False
            logger.info("CPU detected: disabled torch.backends.mkldnn to avoid oneDNN/MKLDNN instability on Windows")
        except Exception as e:
            logger.warning(f"Failed to disable mkldnn backend: {e}")

        try:
            max_threads = max(1, min(4, (os.cpu_count() or 4)))
            torch.set_num_threads(max_threads)
            torch.set_num_interop_threads(1)
            logger.info(f"CPU detected: limiting torch threads to {max_threads} to keep system responsive")
        except Exception as e:
            logger.warning(f"Failed to set torch CPU thread limits: {e}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ── MLflow setup ─────────────────────────────────────────────────────
    import mlflow
    exp_prefix = config.get("mlflow", {}).get("experiment_name_prefix", "")
    experiment_name = f"{exp_prefix}category-encoder"
    setup_experiment(config, experiment_name)
    mlflow_run = mlflow.start_run(run_name=f"run_{datetime.now():%Y%m%d_%H%M%S}")
    run_id = mlflow_run.info.run_id
    logger.info("MLflow run %s started (experiment: %s)", run_id[:8], experiment_name)

    # Merge config defaults with CLI overrides (CLI takes precedence)
    training_params = dict(config.get("models", {}).get("category_encoder", {}))
    training_params.update({
        "model_name": args.model_name,
        "epochs": str(args.epochs),
        "batch_size": str(args.batch_size),
        "eval_batch_size": str(args.eval_batch_size),
        "learning_rate": str(args.learning_rate),
        "max_seq_length": str(args.max_seq_length),
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "device": str(device),
    })
    mlflow.log_params(training_params)

    # Create dataset (use CombinedCategoryDataset for both CSV and PDF)
    logger.info("Creating dataset...")
    
    try:
        if has_pdf:
            # Use combined dataset for CSV + PDF
            dataset = CombinedCategoryDataset(
                csv_path=csv_path if has_csv else None,
                pdf_dir=pdf_dir,
                tokenizer_name=args.model_name,
                max_seq_length=args.max_seq_length,
                is_training=True,
                config=config,
                seed=args.seed
            )
        else:
            # Fall back to CSV-only dataset
            dataset = CategoryDataset(
                data_path=csv_path,
                tokenizer_name=args.model_name,
                max_seq_length=args.max_seq_length,
                is_training=True,
                config=config
            )
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        sys.exit(1)
    
    if len(dataset) == 0:
        logger.error("Dataset is empty after preprocessing")
        sys.exit(1)
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Number of categories: {dataset.num_classes}")
    
    # Split dataset
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    dataset_size = len(dataset)
    train_size = int(args.train_split * dataset_size)
    val_size = int(args.val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    effective_num_workers = args.num_workers
    if os.name == 'nt' and effective_num_workers != 0:
        logger.warning(
            "Windows detected: forcing num_workers=0 to avoid multiprocessing worker import issues. "
            "(You can still run with num_workers=0 reliably.)"
        )
        effective_num_workers = 0
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.get_collate_fn()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.get_collate_fn()
    )
    
    num_classes = dataset.num_classes
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    logger.info(f"Initializing Category Encoder model: {args.model_name}")
    model = CategoryEncoder(
        pretrained_model_name=args.model_name,
        num_classes=num_classes,
        hidden_size=config['models']['category_encoder'].get('hidden_size', 768),
        dropout=config['models']['category_encoder'].get('dropout', 0.1)
    )
    model.to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    ce_cfg = config['models']['category_encoder']
    total_steps = len(train_loader) * args.epochs  # approximate; used for scheduler milestones
    trainer = CategoryEncoderTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=ce_cfg.get('weight_decay', 0.01),
        warmup_steps=ce_cfg.get('warmup_steps', 200),
        device=device,
        use_amp=ce_cfg.get('fp16', False),
        gradient_accumulation_steps=ce_cfg.get('gradient_accumulation_steps', 1),
        total_training_steps=total_steps,
    )

    # Set class weights on the model for imbalanced categories
    counts = dataset.class_counts
    if (counts > 0).all():
        weights = counts.sum() / (counts * len(counts))  # inverse frequency, sum to num_classes
        weights = weights.to(device)
        model.set_class_weights(weights)
        logger.info("Set inverse-frequency class weights on model")
    
    # Get category mapping
    # The category mapping is stored in the dataset during initialization
    # We need to recreate it from the full dataset
    name_to_id = dataset.category_to_idx  # category name -> index
    id_to_name = {idx: name for name, idx in name_to_id.items()}  # index -> category name
    
    logger.info(f"Category mapping loaded: {len(name_to_id)} categories")
    
    # Training loop — best model tracked by macro F1 for robustness on
    # imbalanced categories; early stopping with configurable patience.
    best_val_f1 = 0.0
    best_metrics = {}
    training_history = []
    early_stopping_patience = ce_cfg.get('early_stopping_patience', 0)  # 0 = disabled
    epochs_no_improve = 0
    epochs_completed = 0
    early_stop_triggered = False

    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        epochs_completed = epoch
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 40)

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        logger.info(f"Train - Loss: {train_metrics['avg_loss']:.4f}")

        # Validate
        val_metrics = None
        if val_loader and epoch % args.val_every == 0:
            logger.info("Starting evaluate()...")
            val_metrics = trainer.evaluate(val_loader, id_to_name)
            logger.info("Finished evaluate()")
            logger.info(
                f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
                f"Accuracy: {val_metrics.get('accuracy', 0):.4f}, "
                f"F1 (Macro): {val_metrics.get('f1_macro', 0):.4f}, "
                f"F1 (Weighted): {val_metrics.get('f1_weighted', 0):.4f}"
            )

            # Use macro F1 for best model selection (more robust for
            # imbalanced multi-class than accuracy).
            current_f1 = val_metrics.get('f1_macro', 0)
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                best_metrics = val_metrics.copy()
                epochs_no_improve = 0

                checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
                model.save(checkpoint_path, category_to_idx=dataset.category_to_idx)
                logger.info(f"  >> New best model saved! F1: {best_val_f1:.4f}")
            elif early_stopping_patience > 0:
                epochs_no_improve += 1
                logger.info(
                    f"  No F1 improvement for {epochs_no_improve}/{early_stopping_patience} epochs"
                )
                if epochs_no_improve >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    epochs_no_improve = 0  # reset for the flag check below
                    early_stop_triggered = True

        # Log epoch metrics to MLflow
        epoch_metrics = {
            "train_loss": train_metrics["avg_loss"],
        }
        if val_metrics:
            epoch_metrics.update({
                "val_loss": val_metrics["val_loss"],
                "val_accuracy": val_metrics.get("accuracy", 0),
                "val_f1_macro": val_metrics.get("f1_macro", 0),
                "val_f1_weighted": val_metrics.get("f1_weighted", 0),
                "avg_confidence": val_metrics.get("avg_confidence", 0),
            })
        mlflow.log_metrics(epoch_metrics, step=epoch)

        # Log epoch metrics locally
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_metrics['avg_loss'],
            'val_loss': val_metrics['val_loss'] if val_metrics else None,
            'val_f1_macro': val_metrics.get('f1_macro') if val_metrics else None,
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
    
    # Save final model (B3: persist category_to_idx)
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    model.save(final_path, category_to_idx=dataset.category_to_idx)
    logger.info(f"\nSaved final model to {final_path}")
    
    # Save category mapping
    category_mapping = {
        'id_to_name': {str(k): v for k, v in id_to_name.items()},
        'name_to_id': {v: k for k, v in id_to_name.items()}
    }
    mapping_path = os.path.join(args.output_dir, 'category_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(category_mapping, f, indent=2)
    logger.info(f"Saved category mapping to {mapping_path}")
    
    # Save training summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'args': vars(args),
        'best_val_f1_macro': float(best_val_f1) if isinstance(best_val_f1, (np.floating, np.integer)) else best_val_f1,
        'best_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in best_metrics.items()},
        'epochs_completed': epochs_completed,
        'num_categories': num_classes,
        'category_mapping': category_mapping,
        'training_history': training_history,
        'model_config': {
            'pretrained_model': args.model_name,
            'num_classes': num_classes,
            'max_seq_length': args.max_seq_length
        }
    }
    
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Log MLflow artifacts, register model, and end run ─────────────────
    log_artifacts(args.output_dir, artifact_path="checkpoints")
    mlflow.log_artifact(summary_path, artifact_path="summary")
    mlflow.log_metric("best_val_f1_macro", best_val_f1)
    if best_metrics:
        for k, v in best_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"best_{k}", float(v))

    # Model Registry: log best model with signature and auto-register
    mlflow_cfg = config.get("mlflow", {})
    if mlflow_cfg.get("register_models", False):
        from mlflow.models import infer_signature
        # Build a sample input for signature inference
        sample_input = {
            "input_ids": torch.zeros(1, args.max_seq_length, dtype=torch.long, device=device),
            "attention_mask": torch.ones(1, args.max_seq_length, dtype=torch.long, device=device),
        }
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input["input_ids"], sample_input["attention_mask"])
        sig = infer_signature(sample_input, {"logits": sample_output["logits"].cpu().numpy()})
        reg_name = f"{exp_prefix}category-encoder"
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
    logger.info(f"Best Val F1 (Macro): {best_val_f1:.4f}")
    if best_metrics:
        logger.info(f"Best Val Accuracy: {best_metrics.get('accuracy', 'N/A'):.4f}")
        logger.info(f"Best Val F1 (Weighted): {best_metrics.get('f1_weighted', 'N/A'):.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info("="*60)

    return model


if __name__ == "__main__":
    args = parse_args()
    
    # Quick validation
    if not (args.csv_path or args.data_path or args.pdf_dir):
        print("Error: provide --csv_path (or --data_path for backward compatibility)")
        print("Example: python scripts/train_category.py --csv_path datasets/category_resumes/Resume.csv")
        sys.exit(1)
    
    train(args)
