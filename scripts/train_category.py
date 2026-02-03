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
    trainer = CategoryEncoderTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=config['models']['category_encoder'].get('weight_decay', 0.01),
        warmup_steps=config['models']['category_encoder'].get('warmup_steps', 200),
        device=device
    )
    
    # Get category mapping
    # The category mapping is stored in the dataset during initialization
    # We need to recreate it from the full dataset
    name_to_id = dataset.category_to_idx  # category name -> index
    id_to_name = {idx: name for name, idx in name_to_id.items()}  # index -> category name
    
    logger.info(f"Category mapping loaded: {len(name_to_id)} categories")
    
    # Training loop
    best_val_accuracy = 0
    best_metrics = {}
    training_history = []
    
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60)
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 40)
        
        # Train
        logger.info("Starting train_epoch()...")
        train_metrics = trainer.train_epoch(train_loader)
        logger.info("Finished train_epoch()")
        logger.info(f"Train - Loss: {train_metrics['avg_loss']:.4f}")
        
        # Validate
        val_metrics = None
        if val_loader and epoch % args.val_every == 0:
            logger.info("Starting evaluate()...")
            val_metrics = trainer.evaluate(val_loader, id_to_name)
            logger.info("Finished evaluate()")
            logger.info(f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
                       f"Accuracy: {val_metrics.get('accuracy', 0):.4f}, "
                       f"F1 (Macro): {val_metrics.get('f1_macro', 0):.4f}, "
                       f"F1 (Weighted): {val_metrics.get('f1_weighted', 0):.4f}")
            
            # Save best model
            if val_metrics.get('accuracy', 0) > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_metrics = val_metrics.copy()
                
                # Save best checkpoint
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
                model.save(checkpoint_path)
                logger.info(f"  >> New best model saved! Accuracy: {best_val_accuracy:.4f}")
        
        # Log epoch metrics
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_metrics['avg_loss'],
            'val_loss': val_metrics['val_loss'] if val_metrics else None,
            'val_accuracy': val_metrics.get('accuracy') if val_metrics else None
        }
        training_history.append(epoch_record)
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            metrics_for_checkpoint = val_metrics if val_metrics else train_metrics
            trainer.save_checkpoint(checkpoint_path, epoch, metrics_for_checkpoint)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    model.save(final_path)
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
        'best_val_accuracy': float(best_val_accuracy) if isinstance(best_val_accuracy, (np.floating, np.integer)) else best_val_accuracy,
        'best_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in best_metrics.items()},
        'epochs_completed': args.epochs,
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
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Best Val Accuracy: {best_val_accuracy:.4f}")
    if best_metrics:
        logger.info(f"Best Val F1 (Macro): {best_metrics.get('f1_macro', 'N/A'):.4f}")
        logger.info(f"Best Val F1 (Weighted): {best_metrics.get('f1_weighted', 'N/A'):.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info("="*60)
    
    return model


def evaluate_pretrained(model_path: str, data_path: str, device: str = 'cuda'):
    """Evaluate a pretrained model on test data."""
    logger.info(f"Loading pretrained model from {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    model = CategoryEncoder.load(model_path, device=device)
    
    # Load test data
    test_dataset = CategoryDataset(
        data_path=data_path,
        tokenizer_name=config['models']['category_encoder']['pretrained_model'],
        max_seq_length=config['models']['category_encoder']['max_seq_length'],
        is_training=False,
        config=config
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=True,
        collate_fn=test_dataset.get_collate_fn()
    )
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Get category mapping
    id_to_name = test_dataset.category_to_idx
    
    # Evaluate
    trainer = CategoryEncoderTrainer(model=model, device=device)
    metrics = trainer.evaluate(test_loader, id_to_name)
    
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results:")
    logger.info("="*50)
    for metric, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    logger.info("="*50)
    
    return metrics


if __name__ == "__main__":
    args = parse_args()
    
    # Quick validation
    if not (args.csv_path or args.data_path or args.pdf_dir):
        print("Error: provide --csv_path and/or --pdf_dir (or --data_path for backward compatibility)")
        print("Example: python scripts/train_category.py --csv_path \"datasets/category resumes/Resume/Resume.csv\"")
        sys.exit(1)
    
    train(args)
