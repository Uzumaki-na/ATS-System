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
        help='Use mixed precision training'
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
    
    # Create data loaders using the new optimized loader
    logger.info("Creating data loaders...")
    loaders = create_cross_encoder_loaders(
        train_path=args.train_path,
        val_path=args.val_path if args.val_path else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_length=args.max_seq_length,
        tokenizer_name=args.model_name,
        config=config
    )
    
    if 'train' not in loaders:
        logger.error("Failed to create training data loader")
        sys.exit(1)
    
    train_loader = loaders['train']
    val_loader = loaders.get('val')
    
    logger.info(f"Training batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")
    
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
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = CrossEncoderTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=config['models']['cross_encoder'].get('weight_decay', 0.01),
        warmup_steps=config['models']['cross_encoder'].get('warmup_steps', 500),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        device=device
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = {}
    training_history = []
    
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60)
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 40)
        
        # Train
        train_metrics = trainer.train_epoch(
            train_loader,
            regression_weight=args.regression_weight,
            classification_weight=args.classification_weight
        )
        
        logger.info(f"Train - Loss: {train_metrics['avg_loss']:.4f}, "
                   f"Reg Loss: {train_metrics.get('regression_loss', 0):.4f}, "
                   f"Cls Loss: {train_metrics.get('classification_loss', 0):.4f}")
        
        # Validate
        val_metrics = None
        if val_loader and epoch % args.val_every == 0:
            val_metrics = trainer.evaluate(val_loader)
            logger.info(f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
                       f"MSE: {val_metrics.get('mse', 0):.4f}, "
                       f"MAE: {val_metrics.get('mae', 0):.4f}, "
                       f"Accuracy: {val_metrics.get('accuracy', 0):.4f}, "
                       f"F1 (Macro): {val_metrics.get('f1_macro', 0):.4f}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_metrics = val_metrics.copy()
                
                # Save best checkpoint
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
                model.save(checkpoint_path)
                logger.info(f"  >> New best model saved! Loss: {best_val_loss:.4f}")
        
        # Log epoch metrics
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
        'epochs_completed': args.epochs,
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


def evaluate_pretrained(model_path: str, data_path: str, device: str = 'cuda'):
    """Evaluate a pretrained model on test data."""
    logger.info(f"Loading pretrained model from {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    model = MultiHeadCrossEncoder.load(model_path, device=device)
    
    # Load test data
    schema = config.get('dataset_schemas', {}).get('cross_encoder', {})
    test_dataset = CrossEncoderDataset(
        data_path=data_path,
        tokenizer_name=config['models']['cross_encoder']['pretrained_model'],
        max_seq_length=config['models']['cross_encoder']['max_seq_length'],
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
    
    # Evaluate
    trainer = CrossEncoderTrainer(model=model, device=device)
    metrics = trainer.evaluate(test_loader)
    
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
    if not args.train_path:
        print("Error: --train_path is required")
        print("Example: python scripts/train_cross_encoder.py --train_path datasets/score dataset/train.csv")
        sys.exit(1)
    
    train(args)
