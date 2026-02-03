"""
Data loaders for TriadRank ATS system.
Handles loading and preprocessing of resume-job pairs for training.
Customized for user datasets with optimized data cleaning and preprocessing.
Supports both CSV and PDF resume formats.
"""

import os
import re
import json
import sys
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import spacy
from pathlib import Path
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import configuration
try:
    from config import config
except ImportError:
    # Fallback for when config is not available
    config = {}

# Import PDF processing module
try:
    from .pdf_processing import PDFTextExtractor, PDFResumeDataset, PDFProcessingConfig
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    print("Warning: PDF processing module not available. Install pypdf and pdfplumber.")


# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase for normalization (preserve case for analysis if needed)
    # text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove phone numbers (various formats)
    text = re.sub(r'[\+\(]?[0-9][0-9 \(\)\-]{8,}[0-9]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\-\(\)]', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_resume_text(text: str) -> str:
    """
    Specialized cleaning for resume text.
    
    Args:
        text: Raw resume text
        
    Returns:
        Cleaned resume text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Apply basic text cleaning
    text = clean_text(text)
    
    # Normalize common resume abbreviations
    replacements = {
        r'\byrs?\b': 'years',
        r'\bexp\b': 'experience',
        r'\bsk\b': 'skills',
        r'\bedu\b': 'education',
        r'\bproj\b': 'project',
        r'\bcomp\b': 'company',
        r'\bresp\b': 'responsibilities',
        r'\bachiev\b': 'achievement',
        r'\butiliz\b': 'utilize',
        r'\bdevelop\b': 'development',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Remove multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    
    # Ensure proper spacing around punctuation
    text = re.sub(r'\s*([\.\,\-\(\)])\s*', r'\1 ', text)
    
    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_skills(text: str) -> List[str]:
    """
    Extract potential skills from text using pattern matching.
    
    Args:
        text: Resume or job description text
        
    Returns:
        List of extracted skills
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # Common programming languages
    programming_langs = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go',
        'rust', 'scala', 'kotlin', 'swift', 'php', 'perl', 'r', 'matlab', 'sql',
        'html', 'css', 'bash', 'shell', 'powershell'
    ]
    
    # Common frameworks and libraries
    frameworks = [
        'react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js', 'nodejs',
        'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn', 'sklearn',
        'express', 'fastapi', 'rails', 'laravel', 'bootstrap', 'jquery', 'graphql',
        'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'jenkins', 'terraform'
    ]
    
    # Common soft skills
    soft_skills = [
        'communication', 'leadership', 'teamwork', 'problem solving', 'analytical',
        'project management', 'time management', 'collaboration', 'adaptability'
    ]
    
    text_lower = text.lower()
    
    skills = []
    
    # Check for programming languages
    for lang in programming_langs:
        if re.search(r'\b' + re.escape(lang) + r'\b', text_lower):
            skills.append(lang)
    
    # Check for frameworks
    for fw in frameworks:
        if re.search(r'\b' + re.escape(fw) + r'\b', text_lower):
            skills.append(fw)
    
    # Check for soft skills
    for skill in soft_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            skills.append(skill)
    
    return list(set(skills))


def validate_score(score: Any) -> float:
    """
    Validate and normalize ATS score.
    
    Args:
        score: Raw score value
        
    Returns:
        Normalized score between 0 and 1
    """
    try:
        if pd.isna(score):
            return 0.0
        
        score = float(score)
        
        # If score is already between 0 and 1
        if 0 <= score <= 1:
            return round(score, 4)
        
        # If score is between 0 and 100
        if 0 <= score <= 100:
            return round(score / 100.0, 4)
        
        # If score is on an unknown scale, clip to reasonable range
        return max(0.0, min(1.0, score))
    
    except (ValueError, TypeError):
        return 0.0


def parse_label(label: Any, score: float = 0.0) -> int:
    """
    Parse and normalize label to class index.
    
    Args:
        label: Raw label value
        score: Associated score for fallback
        
    Returns:
        Class index (0: bad_fit, 1: potential_fit, 2: good_fit)
    """
    # Label mapping
    label_map = {
        'good_fit': 2,
        'good fit': 2,
        'good': 2,
        'excellent': 2,
        'best': 2,
        'potential_fit': 1,
        'potential fit': 1,
        'potential': 1,
        'maybe': 1,
        'average': 1,
        'bad_fit': 0,
        'bad fit': 0,
        'bad': 0,
        'poor': 0,
        'reject': 0,
        'no': 0
    }
    
    if pd.isna(label):
        # Fallback to score-based classification
        if score >= 0.75:
            return 2
        elif score >= 0.5:
            return 1
        else:
            return 0
    
    label_str = str(label).lower().strip()
    
    # Direct match
    if label_str in label_map:
        return label_map[label_str]
    
    # Partial match
    for key, value in label_map.items():
        if key in label_str or label_str in key:
            return value
    
    # Final fallback based on score
    if score >= 0.75:
        return 2
    elif score >= 0.5:
        return 1
    else:
        return 0


def validate_category(category: Any) -> str:
    """
    Validate and normalize category name.
    
    Args:
        category: Raw category value
        
    Returns:
        Normalized category name
    """
    if pd.isna(category):
        return "UNKNOWN"
    
    category_str = str(category).strip()
    
    # Handle common variations
    category_normalization = {
        'information-technology': 'INFORMATION-TECHNOLOGY',
        'information technology': 'INFORMATION-TECHNOLOGY',
        'it': 'INFORMATION-TECHNOLOGY',
        'software engineering': 'SOFTWARE-ENGINEERING',
        'software': 'SOFTWARE-ENGINEERING',
        'data science': 'DATA-SCIENCE',
        'data': 'DATA-SCIENCE',
        'hr': 'HUMAN-RESOURCES',
        'human resources': 'HUMAN-RESOURCES',
        'sales': 'SALES',
        'marketing': 'MARKETING',
        'finance': 'FINANCE',
        'accounting': 'ACCOUNTING',
        'healthcare': 'HEALTHCARE',
        'engineering': 'ENGINEERING',
        'design': 'DESIGNER',
        'teaching': 'TEACHER',
        'chef': 'CHEF',
        'aviation': 'AVIATION',
        'banking': 'BANKING',
        'construction': 'CONSTRUCTION',
        'consultant': 'CONSULTANT',
        'public-relations': 'PUBLIC-RELATIONS',
        'public relations': 'PUBLIC-RELATIONS',
        'apparel': 'APPAREL',
        'arts': 'ARTS',
        'agriculture': 'AGRICULTURE',
        'automobile': 'AUTOMOBILE',
        'business-development': 'BUSINESS-DEVELOPMENT',
        'business development': 'BUSINESS-DEVELOPMENT',
        'bpo': 'BPO',
        'fitness': 'FITNESS',
        'legal': 'LEGAL',
        'operations': 'OPERATIONS',
        'project-management': 'PROJECT-MANAGEMENT',
        'project management': 'PROJECT-MANAGEMENT',
        'digital-media': 'DIGITAL-MEDIA',
        'digital media': 'DIGITAL-MEDIA',
    }
    
    normalized = category_normalization.get(category_str.lower(), category_str.upper())
    
    return normalized


# =============================================================================
# DATASET CLASSES
# =============================================================================


def cross_encoder_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for CrossEncoderDataset (top-level for Windows pickling)."""
    # Stack input_ids and attention_mask
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    token_type_ids = None
    if batch and 'token_type_ids' in batch[0]:
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])

    # Get scores and labels
    scores = torch.stack([item['score'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # Keep original texts
    texts = [item['text'] for item in batch]

    collated = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'score': scores,
        'label': labels,
        'text': texts
    }

    if token_type_ids is not None:
        collated['token_type_ids'] = token_type_ids

    return collated

class CrossEncoderDataset(Dataset):
    """
    Dataset for training Model 1 (Cross-Encoder).
    Handles score dataset with resume-job pairs and ATS scores.
    Schema: text, ats_score, original_label
    """
    
    LABEL_MAP = {
        'good_fit': 2,
        'potential_fit': 1,
        'bad_fit': 0
    }
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_seq_length: int = 512,
        is_training: bool = True,
        config: Dict = None
    ):
        """
        Initialize CrossEncoderDataset.
        
        Args:
            data_path: Path to CSV file with score data
            tokenizer_name: Name of pretrained tokenizer
            max_seq_length: Maximum sequence length for tokenization
            is_training: Whether this is training data
            config: Configuration dictionary
        """
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        self.config = config or {}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load and preprocess data
        self.data = self._load_and_preprocess_data()
        
        # Print statistics
        self._print_statistics()
    
    def _load_and_preprocess_data(self) -> List[Dict]:
        """Load data from CSV and apply preprocessing."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Read CSV
        df = pd.read_csv(self.data_path)
        
        # Get column names from config or use defaults
        schema = self.config.get('dataset_schemas', {}).get('cross_encoder', {})
        text_col = schema.get('text_column', 'text')
        score_col = schema.get('score_column', 'ats_score')
        label_col = schema.get('label_column', 'original_label')
        
        # Validate columns exist
        required_cols = [text_col, score_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Try alternative column names
            if 'text' not in df.columns:
                raise ValueError(f"Required columns not found: {required_cols}")
        
        # Preprocess each row
        processed_data = []
        for idx, row in df.iterrows():
            try:
                # Extract and clean text
                text = clean_resume_text(row.get(text_col, row.get('text', '')))
                
                if not text or len(text) < 10:  # Skip very short texts
                    continue
                
                # Validate and normalize score
                score = validate_score(row.get(score_col, 0.0))
                
                # Parse label
                label = parse_label(
                    row.get(label_col, None),
                    score
                )
                
                processed_data.append({
                    'text': text,
                    'score': score,
                    'label': label,
                    'original_label': str(row.get(label_col, '')),
                    'row_idx': idx
                })
            
            except Exception as e:
                # Skip problematic rows
                continue
        
        return processed_data
    
    def _print_statistics(self):
        """Print dataset statistics."""
        if not self.data:
            print(f"[WARNING] No valid data found in {self.data_path}")
            return
        
        scores = [item['score'] for item in self.data]
        labels = [item['label'] for item in self.data]
        
        print(f"\n{'='*60}")
        print(f"CrossEncoderDataset Statistics:")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.data)}")
        print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"Score mean: {np.mean(scores):.4f}")
        print(f"Score std: {np.std(scores):.4f}")
        print(f"Label distribution:")
        label_counts = Counter(labels)
        for label, count in sorted(label_counts.items()):
            label_name = {2: 'good_fit', 1: 'potential_fit', 0: 'bad_fit'}.get(label, 'unknown')
            print(f"  {label_name}: {count} ({100*count/len(labels):.1f}%)")
        print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        item = self.data[idx]

        text = item.get('text', '')
        resume_text = text
        job_description = ''
        if isinstance(text, str) and ' SEP ' in text:
            resume_text, job_description = text.rsplit(' SEP ', 1)

        resume_text = resume_text.strip()
        job_description = job_description.strip()

        try:
            encoding = self.tokenizer(
                resume_text,
                job_description,
                truncation='only_first',
                max_length=self.max_seq_length,
                padding='max_length',
                return_tensors='pt'
            )
        except Exception:
            encoding = self.tokenizer(
                resume_text,
                job_description,
                truncation='longest_first',
                max_length=self.max_seq_length,
                padding='max_length',
                return_tensors='pt'
            )
        
        # Get target values
        score = torch.tensor(item['score'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'score': score,
            'label': label,
            'text': item['text'],
            'row_idx': item.get('row_idx', idx)
        }

        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        return result
    
    def get_collate_fn(self):
        """Return a collate function for batch processing."""
        return cross_encoder_collate_fn


class CategoryDataset(Dataset):
    """
    Dataset for training Model 2 (Category Encoder).
    Handles resume data with job category labels.
    Schema: ID, Resume_str, Resume_html, Category
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_seq_length: int = 256,
        is_training: bool = True,
        config: Dict = None
    ):
        """
        Initialize CategoryDataset.
        
        Args:
            data_path: Path to CSV file with category data
            tokenizer_name: Name of pretrained tokenizer
            max_seq_length: Maximum sequence length for tokenization
            is_training: Whether this is training data
            config: Configuration dictionary
        """
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        self.config = config or {}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load and preprocess data
        self.data = self._load_and_preprocess_data()
        
        # Build category mapping
        self._build_category_mapping()
        
        # Print statistics
        self._print_statistics()
    
    def _load_and_preprocess_data(self) -> List[Dict]:
        """Load data from CSV and apply preprocessing."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Read CSV
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.data_path, encoding='latin-1')
        
        # Get column names from config or use defaults
        schema = self.config.get('dataset_schemas', {}).get('category_encoder', {})
        id_col = schema.get('id_column', 'ID')
        text_col = schema.get('resume_text_column', 'Resume_str')
        html_col = schema.get('resume_html_column', 'Resume_html')
        category_col = schema.get('category_column', 'Category')
        
        # Validate required columns
        required_cols = [text_col, category_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found: {required_cols}. Found: {list(df.columns)}")
        
        # Preprocess each row
        processed_data = []
        for idx, row in df.iterrows():
            try:
                # Extract and clean resume text
                # Prefer Resume_str, fall back to Resume_html if needed
                text = clean_resume_text(row.get(text_col, ''))
                
                if not text or len(text) < 20:  # Skip very short texts
                    continue
                
                # Validate and normalize category
                category = validate_category(row.get(category_col, 'UNKNOWN'))
                
                # Extract skills for additional features
                skills = extract_skills(text)
                
                processed_data.append({
                    'id': str(row.get(id_col, idx)),
                    'text': text,
                    'html_text': clean_resume_text(row.get(html_col, '')) if html_col in row else '',
                    'category': category,
                    'skills': skills,
                    'row_idx': idx
                })
            
            except Exception as e:
                continue
        
        return processed_data
    
    def _build_category_mapping(self):
        """Build category to index mapping."""
        # Get unique categories from data
        categories = sorted(list(set(item['category'] for item in self.data)))
        
        # Add UNKNOWN if not present and there are less than 24 categories
        if len(categories) < 24 and 'UNKNOWN' not in categories:
            categories.append('UNKNOWN')
        
        self.categories = categories
        self.category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        self.num_classes = len(categories)
        
        print(f"Category mapping ({self.num_classes} classes):")
        for cat, idx in self.category_to_idx.items():
            print(f"  {idx}: {cat}")
    
    def _print_statistics(self):
        """Print dataset statistics."""
        if not self.data:
            print(f"[WARNING] No valid data found in {self.data_path}")
            return
        
        categories = [item['category'] for item in self.data]
        category_counts = Counter(categories)
        
        print(f"\n{'='*60}")
        print(f"CategoryDataset Statistics:")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.data)}")
        print(f"Number of categories: {len(self.categories)}")
        print(f"Category distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cat}: {count} ({100*count/len(categories):.1f}%)")
        if len(category_counts) > 10:
            print(f"  ... and {len(category_counts) - 10} more categories")
        print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        item = self.data[idx]
        
        # Tokenize resume text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get category label
        unknown_idx = self.category_to_idx.get('UNKNOWN', 0)
        category_idx = self.category_to_idx.get(item['category'], unknown_idx)
        label = torch.tensor(category_idx, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label,
            'text': item['text'],
            'category': item['category'],
            'skills': item.get('skills', []),
            'row_idx': item.get('row_idx', idx)
        }
    
    def get_collate_fn(self):
        """Return a collate function for batch processing."""
        def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            labels = torch.stack([item['label'] for item in batch])
            
            texts = [item['text'] for item in batch]
            categories = [item['category'] for item in batch]
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': labels,
                'text': texts,
                'category': categories
            }
        
        return collate_fn


class CombinedCategoryDataset(Dataset):
    """
    Combined Dataset for Category Encoder that handles both CSV and PDF resumes.
    
    This dataset can load data from:
    1. CSV file with resume text and categories
    2. Directory with PDF resumes organized by category
    3. Both sources combined
    
    Directory structure for PDF data:
    pdf_dir/
        CATEGORY_NAME_1/
            resume1.pdf
            resume2.pdf
            ...
        CATEGORY_NAME_2/
            resume3.pdf
            resume4.pdf
            ...
    """
    
    def __init__(
        self,
        csv_path: str = None,
        pdf_dir: str = None,
        tokenizer_name: str = "distilbert-base-uncased",
        max_seq_length: int = 256,
        is_training: bool = True,
        config: Dict = None,
        max_samples: int = None,
        seed: int = 42
    ):
        """
        Initialize CombinedCategoryDataset.
        
        Args:
            csv_path: Path to CSV file with resume data (optional)
            pdf_dir: Directory containing PDF resumes by category (optional)
            tokenizer_name: Name of pretrained tokenizer
            max_seq_length: Maximum sequence length for tokenization
            is_training: Whether this is training data
            config: Configuration dictionary
            max_samples: Maximum samples to load (None for all)
            seed: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.pdf_dir = pdf_dir
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        self.config = config or {}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize PDF extractor if PDF directory provided
        self.pdf_extractor = None
        self.pdf_data = []
        if pdf_dir and PDF_PROCESSING_AVAILABLE:
            try:
                pdf_config = PDFProcessingConfig(
                    cache_enabled=True,
                    cache_file=self.config.get('data', {}).get('processed', {}).get('pdf_text_cache', 'data/processed/pdf_text_cache.json')
                )
                self.pdf_extractor = PDFTextExtractor(pdf_config)
                self.pdf_data = self._load_pdf_data(max_samples)
            except Exception as e:
                print(f"Warning: Could not initialize PDF processing: {e}")
                self.pdf_data = []
        
        # Load CSV data
        self.csv_data = []
        if csv_path and os.path.exists(csv_path):
            self.csv_data = self._load_csv_data()
        
        # Combine datasets
        self.data = self._combine_datasets()
        
        # Shuffle if training
        if is_training and len(self.data) > 1:
            import random
            random.seed(seed)
            random.shuffle(self.data)
        
        # Limit samples if specified
        if max_samples is not None:
            self.data = self.data[:max_samples]
        
        # Build category mapping
        self._build_category_mapping()
        
        # Print statistics
        self._print_statistics()
    
    def _load_pdf_data(self, max_samples: int = None) -> List[Dict]:
        """Load and extract text from PDF resumes."""
        if not self.pdf_dir or not os.path.exists(self.pdf_dir):
            return []
        
        pdf_samples = []
        
        # Process each category directory
        for category_dir in sorted(os.listdir(self.pdf_dir)):
            category_path = os.path.join(self.pdf_dir, category_dir)
            
            if not os.path.isdir(category_path):
                continue
            
            # Normalize category name
            category = validate_category(category_dir)
            
            # Process each PDF in this category
            for filename in sorted(os.listdir(category_path)):
                if not filename.lower().endswith('.pdf'):
                    continue
                
                pdf_path = os.path.join(category_path, filename)
                
                # Extract text from PDF
                if self.pdf_extractor:
                    text = self.pdf_extractor.extract(pdf_path)
                else:
                    text = ""
                
                if text and len(text) >= 20:
                    cleaned_text = clean_resume_text(text)
                    
                    pdf_samples.append({
                        'id': os.path.splitext(filename)[0],
                        'text': cleaned_text,
                        'raw_text': text,
                        'category': category,
                        'source': 'pdf',
                        'pdf_path': pdf_path,
                        'filename': filename
                    })
        
        print(f"Loaded {len(pdf_samples)} samples from PDF resumes")
        
        if max_samples and len(pdf_samples) > max_samples:
            pdf_samples = pdf_samples[:max_samples]
        
        return pdf_samples
    
    def _load_csv_data(self) -> List[Dict]:
        """Load data from CSV file."""
        if not self.csv_path or not os.path.exists(self.csv_path):
            return []
        
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            
            schema = self.config.get('dataset_schemas', {}).get('category_encoder', {})
            id_col = schema.get('id_column', 'ID')
            text_col = schema.get('resume_text_column', 'Resume_str')
            category_col = schema.get('category_column', 'Category')
            
            csv_samples = []
            
            for idx, row in df.iterrows():
                text = clean_resume_text(row.get(text_col, ''))
                
                if not text or len(text) < 20:
                    continue
                
                category = validate_category(row.get(category_col, 'UNKNOWN'))
                
                csv_samples.append({
                    'id': str(row.get(id_col, idx)),
                    'text': text,
                    'category': category,
                    'source': 'csv',
                    'row_idx': idx
                })
            
            print(f"Loaded {len(csv_samples)} samples from CSV")
            return csv_samples
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return []
    
    def _combine_datasets(self) -> List[Dict]:
        """Combine CSV and PDF data."""
        combined = []
        
        # Add CSV samples first
        for sample in self.csv_data:
            combined.append(sample)
        
        # Add PDF samples
        for sample in self.pdf_data:
            combined.append(sample)
        
        print(f"Combined dataset: {len(combined)} total samples")
        
        return combined
    
    def _build_category_mapping(self):
        """Build category to index mapping."""
        categories = sorted(list(set(item['category'] for item in self.data)))
        
        # Add UNKNOWN if not present and there are less than 24 categories
        if len(categories) < 24 and 'UNKNOWN' not in categories:
            categories.append('UNKNOWN')
        
        self.categories = categories
        self.category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        self.num_classes = len(categories)
        
        print(f"Category mapping ({self.num_classes} classes):")
        for cat, idx in self.category_to_idx.items():
            print(f"  {idx}: {cat}")
    
    def _print_statistics(self):
        """Print dataset statistics."""
        if not self.data:
            print(f"[WARNING] No valid data found")
            return
        
        # Count by source
        csv_count = sum(1 for item in self.data if item.get('source') == 'csv')
        pdf_count = sum(1 for item in self.data if item.get('source') == 'pdf')
        
        # Count by category
        categories = [item['category'] for item in self.data]
        category_counts = Counter(categories)
        
        print(f"\n{'='*60}")
        print(f"CombinedCategoryDataset Statistics:")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.data)}")
        print(f"  - From CSV: {csv_count}")
        print(f"  - From PDF: {pdf_count}")
        print(f"Number of categories: {len(self.categories)}")
        print(f"Category distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cat}: {count} ({100*count/len(categories):.1f}%)")
        if len(category_counts) > 10:
            print(f"  ... and {len(category_counts) - 10} more categories")
        print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        item = self.data[idx]
        
        # Tokenize resume text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get category label
        unknown_idx = self.category_to_idx.get('UNKNOWN', 0)
        category_idx = self.category_to_idx.get(item['category'], unknown_idx)
        label = torch.tensor(category_idx, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label,
            'text': item['text'],
            'category': item['category'],
            'source': item.get('source', 'unknown'),
            'id': item.get('id', str(idx))
        }
    
    def get_collate_fn(self):
        """Return a collate function for batch processing."""
        def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            labels = torch.stack([item['label'] for item in batch])
            
            texts = [item['text'] for item in batch]
            categories = [item['category'] for item in batch]
            sources = [item.get('source', 'unknown') for item in batch]
            ids = [item.get('id', str(i)) for i, item in enumerate(batch)]
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': labels,
                'text': texts,
                'category': categories,
                'source': sources,
                'id': ids
            }
        
        return collate_fn


class ExtractorDataset(Dataset):
    """
    Dataset for training Model 3 (Extractor & Retrieval).
    Handles resume data with multiple text columns for entity extraction.
    Schema: career_objective, skills, positions, work_experience, education, etc.
    """
    
    def __init__(
        self,
        data_path: str,
        nlp_model: str = "en_core_web_sm",
        is_training: bool = True,
        config: Dict = None
    ):
        """
        Initialize ExtractorDataset.
        
        Args:
            data_path: Path to CSV file with extraction data
            nlp_model: spaCy model name for NER
            is_training: Whether this is training data
            config: Configuration dictionary
        """
        self.data_path = data_path
        self.is_training = is_training
        self.config = config or {}
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            print(f"Downloading spaCy model: {nlp_model}")
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", nlp_model],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to download spaCy model '{nlp_model}'.\n"
                    f"stdout:\n{result.stdout}\n\n"
                    f"stderr:\n{result.stderr}"
                )
            self.nlp = spacy.load(nlp_model)
        
        # Load and preprocess data
        self.data = self._load_and_preprocess_data()
        
        # Print statistics
        self._print_statistics()
    
    def _load_and_preprocess_data(self) -> List[Dict]:
        """Load data from CSV and apply preprocessing."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Read CSV
        df = pd.read_csv(self.data_path, encoding='utf-8')
        
        # Get text columns from config or use defaults
        schema = self.config.get('dataset_schemas', {}).get('extractor', {})
        text_cols = schema.get('text_columns', [
            'career_objective', 'skills', 'work_experience', 'education',
            'positions', 'companies', 'project_details'
        ])
        id_col = schema.get('id_column', 'ID')
        
        # Filter to only columns that exist in the data
        available_cols = [col for col in text_cols if col in df.columns]
        
        # Preprocess each row
        processed_data = []
        for idx, row in df.iterrows():
            try:
                # Combine text from multiple columns
                combined_text_parts = []
                for col in available_cols:
                    text_val = row.get(col, '')
                    if pd.notna(text_val) and str(text_val).strip():
                        cleaned_text = clean_text(str(text_val))
                        if cleaned_text:
                            combined_text_parts.append(cleaned_text)
                
                if not combined_text_parts:
                    continue
                
                combined_text = ' '.join(combined_text_parts)
                
                if len(combined_text) < 30:  # Skip very short texts
                    continue
                
                # Extract sections for structured output
                sections = {}
                for col in available_cols:
                    text_val = row.get(col, '')
                    if pd.notna(text_val) and str(text_val).strip():
                        sections[col] = clean_text(str(text_val))
                
                # Extract skills
                skills = extract_skills(combined_text)
                
                processed_data.append({
                    'id': str(row.get(id_col, idx)),
                    'text': combined_text,
                    'sections': sections,
                    'skills': skills,
                    'row_idx': idx
                })
            
            except Exception as e:
                continue
        
        return processed_data
    
    def _print_statistics(self):
        """Print dataset statistics."""
        if not self.data:
            print(f"[WARNING] No valid data found in {self.data_path}")
            return
        
        text_lengths = [len(item['text'].split()) for item in self.data]
        skills_counts = [len(item.get('skills', [])) for item in self.data]
        
        print(f"\n{'='*60}")
        print(f"ExtractorDataset Statistics:")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.data)}")
        print(f"Word count range: [{min(text_lengths)}, {max(text_lengths)}]")
        print(f"Average word count: {np.mean(text_lengths):.1f}")
        print(f"Average skills per resume: {np.mean(skills_counts):.1f}")
        print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example with entity annotations."""
        item = self.data[idx]
        
        # Process text with spaCy
        doc = self.nlp(item['text'])
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # Extract noun chunks for additional features
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        
        return {
            'id': item['id'],
            'text': item['text'],
            'sections': item.get('sections', {}),
            'entities': entities,
            'tokens': [token.text for token in doc],
            'pos_tags': [token.pos_ for token in doc],
            'ner_tags': [ent.label_ if ent.text else 'O' for ent in doc.ents] + ['O'] * (len(doc) - len(list(doc.ents))),
            'noun_chunks': noun_chunks,
            'skills': item.get('skills', []),
            'row_idx': item.get('row_idx', idx)
        }
    
    def get_collate_fn(self):
        """Return a collate function for batch processing."""
        def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
            ids = [item['id'] for item in batch]
            texts = [item['text'] for item in batch]
            sections = [item.get('sections', {}) for item in batch]
            skills = [item.get('skills', []) for item in batch]
            
            # For sequences, we'll keep them as lists of strings for now
            # Real implementation would need padding and tensor conversion
            tokens = [item.get('tokens', []) for item in batch]
            pos_tags = [item.get('pos_tags', []) for item in batch]
            
            return {
                'id': ids,
                'text': texts,
                'sections': sections,
                'tokens': tokens,
                'pos_tags': pos_tags,
                'skills': skills
            }
        
        return collate_fn


class HardNegativeDataset(Dataset):
    """
    Dataset for hard negative mining.
    Contains pairs of resumes and job descriptions with similarity scores.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_seq_length: int = 512,
        similarity_threshold: float = 0.7,
        config: Dict = None
    ):
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.similarity_threshold = similarity_threshold
        self.config = config or {}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """Load resume-job pairs with similarity scores."""
        if not os.path.exists(self.data_path):
            return []
        
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path, encoding='utf-8')
            data = df.to_dict('records')
        elif self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
        else:
            return []
        
        # Clean and validate data
        cleaned_data = []
        for item in data:
            try:
                resume_text = clean_resume_text(item.get('resume_text', item.get('text', '')))
                if not resume_text or len(resume_text) < 20:
                    continue
                
                cleaned_data.append({
                    'resume_text': resume_text,
                    'job_description': clean_resume_text(item.get('job_description', '')),
                    'similarity': float(item.get('similarity', 0)),
                    'label': str(item.get('label', 'unknown'))
                })
            except Exception:
                continue
        
        # Filter by similarity threshold
        filtered_data = [
            item for item in cleaned_data 
            if item.get('similarity', 0) >= self.similarity_threshold
        ]
        
        return filtered_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a hard negative example."""
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['resume_text'],
            item.get('job_description', ''),
            truncation=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get similarity score
        similarity = torch.tensor(item.get('similarity', 0), dtype=torch.float32)
        
        # Label: hard negative if score is low despite high similarity
        is_hard_negative = 1 if item.get('label', '').lower() in ['bad_fit', 'bad'] else 0
        label = torch.tensor(is_hard_negative, dtype=torch.long)
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'similarity': similarity,
            'label': label,
            'resume_text': item['resume_text'],
            'job_description': item.get('job_description', '')
        }

        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        return result


# =============================================================================
# DATALOADER CREATION FUNCTIONS
# =============================================================================

def create_cross_encoder_loaders(
    train_path: str,
    val_path: str = None,
    batch_size: int = 16,
    num_workers: int = 4,
    max_seq_length: int = 512,
    tokenizer_name: str = "bert-base-uncased",
    config: Dict = None
) -> Dict[str, DataLoader]:
    """
    Create data loaders for Cross-Encoder model.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_seq_length: Maximum sequence length
        tokenizer_name: Name of pretrained tokenizer
        config: Configuration dictionary
        
    Returns:
        Dictionary of data loaders
    """
    dataloaders = {}
    
    # Training loader
    if train_path and os.path.exists(train_path):
        train_dataset = CrossEncoderDataset(
            data_path=train_path,
            tokenizer_name=tokenizer_name,
            max_seq_length=max_seq_length,
            is_training=True,
            config=config
        )
        
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_dataset.get_collate_fn()
        )
    
    # Validation loader
    if val_path and os.path.exists(val_path):
        val_dataset = CrossEncoderDataset(
            data_path=val_path,
            tokenizer_name=tokenizer_name,
            max_seq_length=max_seq_length,
            is_training=False,
            config=config
        )
        
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=val_dataset.get_collate_fn()
        )
    
    return dataloaders


def create_category_loaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_seq_length: int = 256,
    tokenizer_name: str = "distilbert-base-uncased",
    config: Dict = None,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create data loaders for Category Encoder model.
    
    Args:
        data_path: Path to category data
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_split: Proportion of training data
        val_split: Proportion of validation data
        test_split: Proportion of test data
        max_seq_length: Maximum sequence length
        tokenizer_name: Name of pretrained tokenizer
        config: Configuration dictionary
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of data loaders
    """
    if not os.path.exists(data_path):
        return {}
    
    # Load full dataset
    full_dataset = CategoryDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
        is_training=True,
        config=config
    )
    
    # Split dataset
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    dataloaders = {}
    
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=full_dataset.get_collate_fn()
    )
    
    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=full_dataset.get_collate_fn()
    )
    
    if test_size > 0:
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=full_dataset.get_collate_fn()
        )
    
    return dataloaders


def create_extractor_loaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    nlp_model: str = "en_core_web_sm",
    config: Dict = None,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create data loaders for Extractor model.
    
    Args:
        data_path: Path to extraction data
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_split: Proportion of training data
        val_split: Proportion of validation data
        test_split: Proportion of test data
        nlp_model: spaCy model name
        config: Configuration dictionary
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of data loaders
    """
    if not os.path.exists(data_path):
        return {}
    
    # Load full dataset
    full_dataset = ExtractorDataset(
        data_path=data_path,
        nlp_model=nlp_model,
        is_training=True,
        config=config
    )
    
    # Split dataset
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    dataloaders = {}
    
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=full_dataset.get_collate_fn()
    )
    
    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=full_dataset.get_collate_fn()
    )
    
    if test_size > 0:
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=full_dataset.get_collate_fn()
        )
    
    return dataloaders


def create_all_loaders(
    config: Dict = None,
    batch_size: int = None,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create all data loaders based on configuration.
    
    Args:
        config: Configuration dictionary (uses config module if not provided)
        batch_size: Override batch size from config
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary of all data loaders
    """
    # Use provided config or import from config module
    if config is None:
        try:
            from config import config as app_config
            config = app_config
        except ImportError:
            config = {}
    
    # Get batch size from config if not provided
    if batch_size is None:
        batch_size = config.get('models', {}).get('cross_encoder', {}).get('batch_size', 16)
    
    dataloaders = {}
    
    # Get data paths from config
    data_config = config.get('data', {}).get('raw', {})
    
    # Cross-Encoder loaders
    cross_config = data_config.get('cross_encoder', {})
    if os.path.exists(cross_config.get('train', '')):
        cross_loaders = create_cross_encoder_loaders(
            train_path=cross_config.get('train'),
            val_path=cross_config.get('validation'),
            batch_size=batch_size,
            num_workers=num_workers,
            max_seq_length=config.get('models', {}).get('cross_encoder', {}).get('max_seq_length', 512),
            tokenizer_name=config.get('models', {}).get('cross_encoder', {}).get('pretrained_model', 'bert-base-uncased'),
            config=config
        )
        dataloaders.update(cross_loaders)
    
    # Category Encoder loaders
    if os.path.exists(data_config.get('category_resumes', '')):
        cat_loaders = create_category_loaders(
            data_path=data_config.get('category_resumes'),
            batch_size=config.get('models', {}).get('category_encoder', {}).get('batch_size', 32),
            num_workers=num_workers,
            max_seq_length=config.get('models', {}).get('category_encoder', {}).get('max_seq_length', 256),
            tokenizer_name=config.get('models', {}).get('category_encoder', {}).get('pretrained_model', 'distilbert-base-uncased'),
            config=config
        )
        dataloaders.update(cat_loaders)
    
    # Extractor loaders
    if os.path.exists(data_config.get('extraction_pairs', '')):
        ext_loaders = create_extractor_loaders(
            data_path=data_config.get('extraction_pairs'),
            batch_size=config.get('models', {}).get('extractor', {}).get('batch_size', 32),
            num_workers=num_workers,
            nlp_model=config.get('models', {}).get('extractor', {}).get('spacy_model', 'en_core_web_sm'),
            config=config
        )
        dataloaders.update(ext_loaders)
    
    return dataloaders


# =============================================================================
# INFERENCE UTILITIES
# =============================================================================

def load_inference_batch(
    resumes: List[str],
    job_description: str,
    tokenizer_name: str = "bert-base-uncased",
    max_seq_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Create a batch for inference.
    
    Args:
        resumes: List of resume texts
        job_description: Job description text
        tokenizer_name: Name of pretrained tokenizer
        max_seq_length: Maximum sequence length
        
    Returns:
        Tokenized batch ready for model inference
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    cleaned_job = clean_resume_text(job_description)
    cleaned_resumes = [clean_resume_text(resume) for resume in resumes]
    job_descriptions = [cleaned_job] * len(cleaned_resumes)

    try:
        encoding = tokenizer(
            cleaned_resumes,
            job_descriptions,
            truncation='only_first',
            max_length=max_seq_length,
            padding=True,
            return_tensors='pt'
        )
    except Exception:
        encoding = tokenizer(
            cleaned_resumes,
            job_descriptions,
            truncation='longest_first',
            max_length=max_seq_length,
            padding=True,
            return_tensors='pt'
        )

    result = {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }

    if 'token_type_ids' in encoding:
        result['token_type_ids'] = encoding['token_type_ids']

    return result


def prepare_resume_for_inference(
    resume_text: str,
    job_description: str = None
) -> str:
    """
    Prepare a single resume for model inference.
    
    Args:
        resume_text: Raw resume text
        job_description: Optional job description for cross-encoder
        
    Returns:
        Processed text ready for tokenization
    """
    # Clean the resume text
    cleaned_text = clean_resume_text(resume_text)
    
    # If job description is provided, combine for cross-encoder
    if job_description:
        cleaned_job = clean_resume_text(job_description)
        combined_text = f"{cleaned_text} SEP {cleaned_job}"
        return combined_text
    
    return cleaned_text


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("Testing data loaders with user datasets...")
    print("=" * 60)
    
    # Test paths - adjust these paths as needed
    base_path = "datasets_extracted/reumse-job datasets"
    
    try:
        # Test Cross-Encoder Dataset
        print("\n1. Testing CrossEncoderDataset...")
        cross_train_path = f"{base_path}/score dataset/train.csv"
        if os.path.exists(cross_train_path):
            cross_dataset = CrossEncoderDataset(
                data_path=cross_train_path,
                max_seq_length=512,
                config={'dataset_schemas': {'cross_encoder': {
                    'text_column': 'text',
                    'score_column': 'ats_score',
                    'label_column': 'original_label'
                }}}
            )
            print(f"   Loaded {len(cross_dataset)} cross-encoder samples")
        else:
            print(f"   File not found: {cross_train_path}")
        
        # Test Category Dataset
        print("\n2. Testing CategoryDataset...")
        category_path = f"{base_path}/category resumes/Resume/Resume.csv"
        if os.path.exists(category_path):
            category_dataset = CategoryDataset(
                data_path=category_path,
                max_seq_length=256,
                config={'dataset_schemas': {'category_encoder': {
                    'id_column': 'ID',
                    'resume_text_column': 'Resume_str',
                    'category_column': 'Category'
                }}}
            )
            print(f"   Loaded {len(category_dataset)} category samples")
            print(f"   Number of categories: {category_dataset.num_classes}")
        else:
            print(f"   File not found: {category_path}")
        
        # Test Extractor Dataset
        print("\n3. Testing ExtractorDataset...")
        extraction_path = f"{base_path}/bi encoder retrieval dataset/resume_data_for_ranking.csv"
        if os.path.exists(extraction_path):
            extractor_dataset = ExtractorDataset(
                data_path=extraction_path,
                config={'dataset_schemas': {'extractor': {
                    'text_columns': ['career_objective', 'skills', 'work_experience', 'education'],
                    'id_column': 'ID'
                }}}
            )
            print(f"   Loaded {len(extractor_dataset)} extraction samples")
        else:
            print(f"   File not found: {extraction_path}")
        
        print("\n" + "=" * 60)
        print("Data loader testing complete!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
