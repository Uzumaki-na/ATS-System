"""
Training Script for Model 3: Entity Extractor & Hard Negative Miner
Uses spaCy for NER with custom training.
Customized for user datasets with optimized data loading.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import spacy
from spacy.training.example import Example

from config import config
from models.extractor import ResumeExtractor, HardNegativeMiner
from data.loaders import ExtractorDataset, create_extractor_loaders


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Entity Extractor')
    
    # Data path from config
    data_config = config.get('data', {}).get('raw', {})
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=data_config.get('extraction_pairs', ''),
        help='Path to extraction data CSV'
    )
    parser.add_argument(
        '--spacy_model',
        type=str,
        default=config['models']['extractor'].get('spacy_model', 'en_core_web_sm'),
        help='Base spaCy model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=config['models']['extractor'].get('epochs', 50),
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config['models']['extractor'].get('batch_size', 32),
        help='Training batch size'
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=config['models']['extractor'].get('eval_batch_size', 64),
        help='Evaluation batch size'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/extractor',
        help='Output directory for model'
    )
    parser.add_argument(
        '--similarity_threshold',
        type=float,
        default=config.get('hard_negative_mining', {}).get('similarity_threshold', 0.7),
        help='Similarity threshold for hard negative mining'
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
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def setup_spacy_model(spacy_model: str):
    """Setup and return spaCy model, downloading if necessary."""
    try:
        nlp = spacy.load(spacy_model)
        logger.info(f"Loaded spaCy model: {spacy_model}")
        return nlp
    except OSError:
        logger.info(f"Downloading spaCy model: {spacy_model}")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", spacy_model])
        nlp = spacy.load(spacy_model)
        logger.info(f"Downloaded and loaded spaCy model: {spacy_model}")
        return nlp


def prepare_spacy_training_data(dataset: ExtractorDataset) -> list:
    """
    Prepare training data for spaCy NER from the dataset.
    
    Args:
        dataset: ExtractorDataset instance
        
    Returns:
        List of (text, {'entities': [(start, end, label), ...]}) tuples
    """
    training_data = []
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        text = item['text']
        entities = item.get('entities', [])
        
        if entities:
            # Convert entities to spaCy format
            spacy_entities = []
            for ent in entities:
                # Ensure entities are within text bounds
                if ent['start'] >= 0 and ent['end'] <= len(text):
                    spacy_entities.append((
                        ent['start'],
                        ent['end'],
                        ent['label']
                    ))
            
            if spacy_entities:
                training_data.append((text, {'entities': spacy_entities}))
    
    logger.info(f"Prepared {len(training_data)} training examples with entities")
    return training_data


def train_extractor(args, training_data: list):
    """
    Train entity extractor using spaCy.
    
    Args:
        args: Command line arguments
        training_data: List of (text, {'entities': [...]}) tuples
    """
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base model
    logger.info(f"Loading spaCy model: {args.spacy_model}")
    nlp = setup_spacy_model(args.spacy_model)
    
    # Get the NER pipe
    if 'ner' not in nlp.pipe_names:
        nlp.add_pipe('ner')
    
    ner = nlp.get_pipe('ner')
    
    # Define entity labels from config
    entity_config = config.get('entity_types', {})
    
    # Build comprehensive entity labels
    entity_labels = set()
    
    # Add labels from config
    for category, entities in entity_config.items():
        if isinstance(entities, list):
            for entity in entities:
                entity_labels.add(entity.upper())
    
    # Add common entity types for resumes
    common_labels = [
        'SKILL', 'PROGRAMMING_LANGUAGE', 'FRAMEWORK', 'DATABASE', 'TOOL',
        'COMPANY', 'JOB_TITLE', 'DURATION', 'DESCRIPTION',
        'DEGREE', 'INSTITUTION', 'FIELD_OF_STUDY', 'GRADUATION_DATE',
        'NAME', 'EMAIL', 'PHONE', 'LOCATION', 'LINKEDIN',
        'PROJECT_NAME', 'PROJECT_DESCRIPTION', 'ACHIEVEMENT'
    ]
    
    for label in common_labels:
        entity_labels.add(label)
    
    # Add labels to NER
    for label in entity_labels:
        if label not in ner.labels:
            ner.add_label(label)
    
    logger.info(f"Entity labels: {list(ner.labels)}")
    
    # Training configuration
    n_iter = args.epochs
    batch_size = args.batch_size
    
    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    
    logger.info(f"Starting training for {n_iter} epochs...")
    logger.info(f"Training samples: {len(training_data)}")
    logger.info(f"Batch size: {batch_size}")
    
    examples = []
    for text, annotations in training_data:
        try:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotations))
        except Exception:
            continue

    if not examples:
        logger.error("No valid spaCy training examples could be created")
        sys.exit(1)

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()

        # Training loop
        for epoch in range(n_iter):
            random.shuffle(examples)

            losses = {}
            epoch_losses = []

            # Create batches
            batch_count = 0
            for i in range(0, len(examples), batch_size):
                batch = examples[i:i + batch_size]

                if len(batch) == 0:
                    continue

                before = losses.get('ner', 0.0)
                nlp.update(
                    batch,
                    sgd=optimizer,
                    losses=losses
                )

                epoch_losses.append(losses.get('ner', 0.0) - before)
                batch_count += 1

            # Log progress
            if epoch % 10 == 0 or epoch == n_iter - 1:
                avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                logger.info(f"Epoch {epoch + 1}/{n_iter}, Avg Loss: {avg_loss:.4f}, Batches: {batch_count}")
    
    # Save the model
    model_path = os.path.join(args.output_dir, 'extractor_model')
    nlp.to_disk(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save entity labels
    labels_path = os.path.join(args.output_dir, 'entity_labels.json')
    with open(labels_path, 'w') as f:
        json.dump(list(ner.labels), f, indent=2)
    logger.info(f"Saved entity labels to {labels_path}")
    
    return nlp


def evaluate_extractor(nlp, test_data: list):
    """
    Evaluate the trained extractor.
    
    Args:
        nlp: Trained spaCy model
        test_data: List of (text, {'entities': [...]}) tuples
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating extractor...")
    
    if not test_data:
        logger.warning("No test data provided for evaluation")
        return {'precision': 0, 'recall': 0, 'f1': 0, 'entity_accuracy': 0}
    
    correct = 0
    total_true = 0
    total_predicted = 0
    
    for text, annotations in test_data:
        doc = nlp(text)
        predicted_entities = set()
        
        for ent in doc.ents:
            predicted_entities.add((ent.start_char, ent.end_char, ent.label_))
        
        true_entities = set()
        for ent in annotations.get('entities', []):
            true_entities.add((ent['start'], ent['end'], ent['label']))
        
        correct += len(predicted_entities & true_entities)
        total_true += len(true_entities)
        total_predicted += len(predicted_entities)
    
    precision = correct / total_predicted if total_predicted > 0 else 0
    recall = correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'entity_accuracy': precision  # Using precision as entity accuracy
    }
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return metrics


def train_hard_negative_miner(args):
    """
    Train hard negative miner configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        HardNegativeMiner instance
    """
    logger.info("Initializing Hard Negative Miner...")
    
    miner = HardNegativeMiner(
        similarity_threshold=args.similarity_threshold,
        max_negatives_per_positive=config.get('hard_negative_mining', {}).get('max_negatives_per_positive', 5)
    )
    
    # Save miner configuration
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, 'miner_config.json')
    
    miner_config = {
        'similarity_threshold': args.similarity_threshold,
        'max_negatives_per_positive': config.get('hard_negative_mining', {}).get('max_negatives_per_positive', 5),
        'mining_batch_size': config.get('hard_negative_mining', {}).get('mining_batch_size', 1000),
        'use_gpu': config.get('hard_negative_mining', {}).get('use_gpu', True)
    }
    
    with open(config_path, 'w') as f:
        json.dump(miner_config, f, indent=2)
    
    logger.info(f"Saved miner config to {config_path}")
    
    return miner


def test_extraction(args, nlp=None):
    """Test the extractor on sample data."""
    logger.info("Testing extraction...")
    
    # Initialize extractor (use trained model if provided)
    extractor = ResumeExtractor(spacy_model=args.spacy_model)
    if nlp is not None:
        extractor.nlp = nlp
    
    # Sample resume text
    sample_resume = """
    John Doe
    Software Engineer at Google
    
    Skills: Python, Java, TensorFlow, Docker, Kubernetes, AWS, React
    
    Experience:
    - Senior Software Engineer at Google (2020-Present)
    - Software Engineer at Microsoft (2018-2020)
    
    Education:
    - M.S. Computer Science, Stanford University (2018)
    - B.S. Computer Science, MIT (2016)
    
    Projects:
    - Built scalable ML pipeline using TensorFlow and Kubernetes
    - Developed web application using React and Node.js
    """
    
    # Extract entities
    result = extractor.extract(sample_resume)
    
    logger.info("\n" + "="*50)
    logger.info("Extraction Results:")
    logger.info("="*50)
    logger.info(f"Skills: {result.get('skills', [])}")
    logger.info(f"Education: {result.get('education', [])}")
    logger.info(f"Experience: {result.get('experience', [])}")
    logger.info(f"Raw Entities: {result.get('raw_entities', [])}")
    logger.info("="*50)
    
    # Test skill overlap
    job_skills = ['python', 'machine learning', 'docker', 'react']
    resume_skills = extractor.extract_skills_list(sample_resume)
    
    overlap = extractor.get_skill_overlap(resume_skills, job_skills)
    logger.info(f"\nSkill Overlap Analysis:")
    logger.info(f"Resume skills: {resume_skills}")
    logger.info(f"Job skills: {job_skills}")
    logger.info(f"Overlap: {overlap}")
    
    return result


def main(args):
    """Main training function."""
    logger.info("="*60)
    logger.info("Extractor Training Pipeline")
    logger.info("="*60)
    
    # Validate data path
    if not args.data_path or not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log data path
    logger.info(f"Data file: {args.data_path}")
    
    # Create data loaders using the new optimized loader
    logger.info("Creating data loaders...")
    loaders = create_extractor_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
        nlp_model=args.spacy_model,
        config=config,
        seed=args.seed
    )
    
    if 'train' not in loaders:
        logger.error("Failed to create training data loader")
        sys.exit(1)
    
    train_loader = loaders['train']
    val_loader = loaders.get('val')
    
    logger.info(f"Training batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")
    
    # Prepare spaCy training data
    logger.info("Preparing spaCy training data...")
    
    # Get full dataset for training
    full_dataset = ExtractorDataset(
        data_path=args.data_path,
        nlp_model=args.spacy_model,
        is_training=True,
        config=config
    )
    
    # Convert to spaCy format
    training_data = prepare_spacy_training_data(full_dataset)
    
    if not training_data:
        logger.warning("No entity annotations found in training data. Creating synthetic training data...")
        # Fall back to extracting entities using existing NER
        for idx in range(min(100, len(full_dataset))):
            item = full_dataset[idx]
            text = item['text']
            
            # Use spaCy to extract entities
            doc = full_dataset.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_
                })
            
            if entities:
                training_data.append((text, {'entities': entities}))
    
    if not training_data:
        logger.error("No training data available. Exiting.")
        sys.exit(1)
    
    logger.info(f"Training samples with entities: {len(training_data)}")
    
    # Train entity extractor
    logger.info("\n" + "="*60)
    logger.info("Training Entity Extractor")
    logger.info("="*60)
    nlp = train_extractor(args, training_data)
    
    # Evaluate on validation data
    val_metrics = {}
    if val_loader:
        logger.info("\n" + "="*60)
        logger.info("Evaluating on Validation Data")
        logger.info("="*60)
        
        # Get validation data in spaCy format
        val_indices = val_loader.dataset.indices
        val_data = []
        for idx in val_indices:
            item = full_dataset[idx]
            if item.get('entities'):
                val_data.append((item['text'], {'entities': [
                    {'start': e['start'], 'end': e['end'], 'label': e['label']}
                    for e in item['entities']
                ]}))
        
        if val_data:
            val_metrics = evaluate_extractor(nlp, val_data)
    
    # Train hard negative miner
    logger.info("\n" + "="*60)
    logger.info("Training Hard Negative Miner")
    logger.info("="*60)
    miner = train_hard_negative_miner(args)
    
    # Test extraction
    logger.info("\n" + "="*60)
    logger.info("Testing Extraction")
    logger.info("="*60)
    test_extraction(args, nlp)
    
    # Save training summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'args': vars(args),
        'spacy_model': args.spacy_model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'similarity_threshold': args.similarity_threshold,
        'training_samples': len(training_data),
        'validation_metrics': val_metrics,
        'model_path': os.path.join(args.output_dir, 'extractor_model')
    }
    
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Model saved to: {os.path.join(args.output_dir, 'extractor_model')}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("="*60)
    
    return nlp, miner


if __name__ == "__main__":
    args = parse_args()
    
    # Quick validation
    if not args.data_path:
        print("Error: --data_path is required")
        print("Example: python scripts/train_extractor.py --data_path datasets/bi encoder retrieval dataset/resume_data_for_ranking.csv")
        sys.exit(1)
    
    main(args)
