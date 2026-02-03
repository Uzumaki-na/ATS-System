"""
PDF Text Extraction Module for Resume Processing.
Handles extraction of text from PDF resumes for real-world ATS scenarios.
"""

import os
import json
import warnings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import hashlib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import PDF libraries, provide fallbacks if not available
PDF_LIBS_AVAILABLE = {}


try:
    import pypdf
    PDF_LIBS_AVAILABLE['pypdf'] = True
except ImportError:
    PDF_LIBS_AVAILABLE['pypdf'] = False

try:
    import pdfplumber
    PDF_LIBS_AVAILABLE['pdfplumber'] = True
except ImportError:
    PDF_LIBS_AVAILABLE['pdfplumber'] = False


@dataclass
class PDFProcessingConfig:
    """Configuration for PDF processing."""
    enabled: bool = True
    extract_images: bool = False
    min_text_length: int = 50
    cache_enabled: bool = True
    cache_file: str = "data/processed/pdf_text_cache.json"
    extraction_method: str = "auto"  # pypdf, pdfplumber, auto


class PDFTextExtractor:
    """
    Extract text from PDF resumes using available libraries.
    
    Supports multiple extraction methods:
    - pdfplumber: Better layout preservation, tables
    - pypdf: Simple, reliable text extraction
    """
    
    def __init__(self, config: PDFProcessingConfig = None):
        """
        Initialize PDF text extractor.
        
        Args:
            config: PDF processing configuration
        """
        self.config = config or PDFProcessingConfig()
        self.cache = {}
        
        # Load cache if exists
        if self.config.cache_enabled and os.path.exists(self.config.cache_file):
            try:
                with open(self.config.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content to use as cache key."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return file_path
    
    def _save_cache(self):
        """Save cache to disk."""
        if self.config.cache_enabled:
            os.makedirs(os.path.dirname(self.config.cache_file), exist_ok=True)
            try:
                with open(self.config.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: Could not save PDF cache: {e}")
    
    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """
        Extract text using pdfplumber (better for structured layouts).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text string
        """
        try:
            import pdfplumber
            
            text_parts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Extract text with layout preservation
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if page_text:
                        text_parts.append(page_text)
                    
                    # Optionally extract tables
                    if self.config.extract_images:
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                table_text = self._format_table(table)
                                text_parts.append(table_text)
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            print(f"Error with pdfplumber: {e}")
            return ""
    
    def extract_with_pypdf(self, pdf_path: str) -> str:
        """
        Extract text using pypdf (simple and reliable).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text string
        """
        try:
            from pypdf import PdfReader
            
            text_parts = []
            reader = PdfReader(pdf_path)
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            print(f"Error with pypdf: {e}")
            return ""
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format extracted table as text."""
        if not table:
            return ""
        
        formatted_lines = []
        for row in table:
            if row:
                # Clean cell values
                cells = [str(cell).strip() if cell else "" for cell in row]
                # Join with proper spacing
                line = ' | '.join(cells)
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def extract(self, pdf_path: str, use_cache: bool = True) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            use_cache: Whether to use cached results
            
        Returns:
            Extracted text string
        """
        if not os.path.exists(pdf_path):
            return ""
        
        # Check cache
        if use_cache and self.config.cache_enabled:
            file_hash = self._get_file_hash(pdf_path)
            if file_hash in self.cache:
                return self.cache[file_hash]
        
        # Extract text
        extracted_text = ""
        
        # Try extraction methods in order of preference
        if self.config.extraction_method == "auto" or self.config.extraction_method == "pdfplumber":
            if PDF_LIBS_AVAILABLE.get('pdfplumber', False):
                extracted_text = self.extract_with_pdfplumber(pdf_path)
        
        # Fallback to pypdf if pdfplumber failed or not available
        if not extracted_text:
            if PDF_LIBS_AVAILABLE.get('pypdf', False):
                extracted_text = self.extract_with_pypdf(pdf_path)
            elif self.config.extraction_method == "pdfplumber":
                # If pdfplumber was requested but not available, try pypdf
                if PDF_LIBS_AVAILABLE.get('pypdf', False):
                    extracted_text = self.extract_with_pypdf(pdf_path)
        
        # Validate extraction
        if len(extracted_text.strip()) < self.config.min_text_length:
            print(f"Warning: Low text extraction for {pdf_path}: {len(extracted_text)} chars")
        
        # Save to cache
        if use_cache and self.config.cache_enabled and extracted_text:
            file_hash = self._get_file_hash(pdf_path)
            self.cache[file_hash] = extracted_text
            self._save_cache()
        
        return extracted_text
    
    def extract_batch(
        self, 
        pdf_paths: List[str], 
        show_progress: bool = True
    ) -> Dict[str, str]:
        """
        Extract text from multiple PDF files.
        
        Args:
            pdf_paths: List of PDF file paths
            show_progress: Show progress indicator
            
        Returns:
            Dictionary mapping file paths to extracted text
        """
        results = {}
        total = len(pdf_paths)
        
        for idx, pdf_path in enumerate(pdf_paths):
            if show_progress and (idx + 1) % 10 == 0:
                print(f"Processing PDF {idx + 1}/{total}: {os.path.basename(pdf_path)}")
            
            text = self.extract(pdf_path)
            if text:
                results[pdf_path] = text
        
        return results
    
    def clear_cache(self):
        """Clear the text extraction cache."""
        self.cache = {}
        if os.path.exists(self.config.cache_file):
            os.remove(self.config.cache_file)
        print("PDF cache cleared.")


class PDFResumeDataset:
    """
    Dataset class for handling PDF resumes organized by category.
    
    Expects directory structure:
    base_dir/
        CATEGORY_NAME_1/
            resume1.pdf
            resume2.pdf
            ...
        CATEGORY_NAME_2/
            resume3.pdf
            resume4.pdf
            ...
        ...
    """
    
    def __init__(
        self,
        pdf_dir: str,
        config: PDFProcessingConfig = None,
        max_samples: int = None,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize PDF Resume Dataset.
        
        Args:
            pdf_dir: Base directory containing category subdirectories
            config: PDF processing configuration
            max_samples: Maximum samples to load (None for all)
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
        """
        self.pdf_dir = pdf_dir
        self.config = config or PDFProcessingConfig()
        self.extractor = PDFTextExtractor(self.config)
        
        # Collect all PDFs with their categories
        self.samples = self._collect_samples()
        
        # Shuffle if needed
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(self.samples)
        
        # Limit samples if specified
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
    
    def _collect_samples(self) -> List[Dict[str, str]]:
        """Collect all PDF files with their category labels."""
        samples = []
        
        if not os.path.exists(self.pdf_dir):
            print(f"Warning: PDF directory not found: {self.pdf_dir}")
            return samples
        
        # Get all subdirectories (each is a category)
        for category_dir in sorted(os.listdir(self.pdf_dir)):
            category_path = os.path.join(self.pdf_dir, category_dir)
            
            if not os.path.isdir(category_path):
                continue
            
            # Get all PDF files in this category
            for filename in sorted(os.listdir(category_path)):
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(category_path, filename)
                    samples.append({
                        'pdf_path': pdf_path,
                        'category': category_dir,
                        'filename': filename,
                        'id': os.path.splitext(filename)[0]
                    })
        
        print(f"Collected {len(samples)} PDF resumes from {len(set(s['category'] for s in samples))} categories")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single PDF resume sample."""
        sample = self.samples[idx]
        
        # Extract text from PDF
        text = self.extractor.extract(sample['pdf_path'])
        
        # Clean extracted text
        from .loaders import clean_resume_text
        cleaned_text = clean_resume_text(text)
        
        return {
            'id': sample['id'],
            'text': cleaned_text,
            'category': sample['category'],
            'pdf_path': sample['pdf_path'],
            'filename': sample['filename'],
            'raw_text': text
        }
    
    def extract_all(self, show_progress: bool = True) -> Dict[str, str]:
        """
        Extract text from all PDFs in the dataset.
        
        Args:
            show_progress: Show progress indicator
            
        Returns:
            Dictionary mapping PDF paths to extracted text
        """
        pdf_paths = [s['pdf_path'] for s in self.samples]
        return self.extractor.extract_batch(pdf_paths, show_progress)
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get the distribution of samples across categories."""
        from collections import Counter
        categories = [s['category'] for s in self.samples]
        return dict(Counter(categories))


def load_pdf_resumes_for_training(
    pdf_dir: str,
    csv_path: str = None,
    max_samples: int = None,
    config: PDFProcessingConfig = None
) -> List[Dict[str, Any]]:
    """
    Load PDF resumes for training, optionally matching with CSV data.
    
    Args:
        pdf_dir: Directory containing PDF resumes by category
        csv_path: Optional CSV file to match PDFs with
        max_samples: Maximum samples to load
        config: PDF processing configuration
        
    Returns:
        List of sample dictionaries with text and labels
    """
    import pandas as pd
    
    # Load PDF dataset
    pdf_dataset = PDFResumeDataset(
        pdf_dir=pdf_dir,
        config=config,
        max_samples=max_samples,
        shuffle=True
    )
    
    # Load CSV data if provided
    csv_data = {}
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # Match by ID or filename
            sample_id = str(row.get('ID', ''))
            csv_data[sample_id] = {
                'id': sample_id,
                'text': row.get('Resume_str', ''),
                'category': row.get('Category', ''),
                'html': row.get('Resume_html', '')
            }
    
    # Combine PDF and CSV data
    combined_samples = []
    
    for sample in pdf_dataset:
        # Try to find matching CSV entry
        matching_csv = csv_data.get(sample['id'], {})
        
        # Use PDF text, fall back to CSV text
        text = sample['text'] if len(sample['text']) > len(matching_csv.get('text', '')) else matching_csv.get('text', '')
        
        if text and len(text) >= 50:  # Minimum text length
            combined_samples.append({
                'id': sample['id'],
                'text': text,
                'category': sample['category'] or matching_csv.get('category', 'UNKNOWN'),
                'source': 'pdf' if sample['text'] else 'csv',
                'filename': sample['filename']
            })
    
    print(f"Combined dataset: {len(combined_samples)} samples from PDFs and CSV")
    
    return combined_samples


# Installation helper
def check_pdf_libs():
    """Check which PDF libraries are available."""
    print("\n" + "="*50)
    print("PDF Library Availability:")
    print("="*50)
    
    libs = [
        ("pypdf", "Simple, reliable PDF text extraction"),
        ("pdfplumber", "Advanced PDF extraction with layout support")
    ]
    
    for lib_name, description in libs:
        available = PDF_LIBS_AVAILABLE.get(lib_name, False)
        status = "✓ Installed" if available else "✗ Not installed"
        print(f"{lib_name:15} {status:15} - {description}")
    
    print("\nTo install missing libraries:")
    print("  pip install pypdf pdfplumber")
    print("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Resume Processing")
    parser.add_argument("--pdf_dir", type=str, help="Directory containing PDF resumes")
    parser.add_argument("--check", action="store_true", help="Check available libraries")
    parser.add_argument("--extract", type=str, help="Extract text from a single PDF")
    
    args = parser.parse_args()
    
    if args.check:
        check_pdf_libs()
    elif args.extract:
        extractor = PDFTextExtractor()
        text = extractor.extract(args.extract)
        print(f"Extracted {len(text)} characters")
        print(text[:500] + "..." if len(text) > 500 else text)
    elif args.pdf_dir:
        print(f"Scanning PDF directory: {args.pdf_dir}")
        dataset = PDFResumeDataset(args.pdf_dir)
        print(f"Found {len(dataset)} resumes")
        dist = dataset.get_category_distribution()
        print("\nCategory distribution:")
        for cat, count in sorted(dist.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cat}: {count}")
