"""
Model 3: Entity Extractor for Resume Parsing
Extracts skills, experience, education, and other entities from resumes.
Also handles hard negative mining for training.
"""

import torch
import torch.nn as nn
import spacy
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import config


class ResumeExtractor:
    """
    Entity extraction module for parsing resumes.
    Uses spaCy NER with custom rules for resume-specific entities.
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_custom_rules: bool = True
    ):
        """
        Initialize the extractor with spaCy model.
        
        Args:
            spacy_model: spaCy model name to load
            enable_custom_rules: Whether to enable custom extraction rules
        """
        self.nlp = spacy.load(spacy_model)
        self.enable_custom_rules = enable_custom_rules
        
        # Custom skill patterns
        self.skill_patterns = self._build_skill_patterns()
        
        # Company name patterns
        self.company_patterns = self._build_company_patterns()
        
        # Degree patterns
        self.degree_patterns = self._build_degree_patterns()
    
    def _build_skill_patterns(self) -> Dict:
        """Build regex patterns for skill extraction."""
        return {
            'programming_language': [
                r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|scala|kotlin|swift|php|perl|r)\b',
                r'\b(pandas|numpy|scipy|matplotlib|tensorflow|pytorch|keras)\b'
            ],
            'framework': [
                r'\b(django|flask|fastapi|spring|react|angular|vue|node\.js|express)\b',
                r'\b(laravel|symfony|rails|asp\.net|next\.js|nuxt)\b'
            ],
            'database': [
                r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|oracle|sqlite)\b',
                r'\b(cassandra|dynamodb|firebase|postgis|neo4j)\b'
            ],
            'tool': [
                r'\b(git|docker|kubernetes|jenkins|aws|azure|gcp|terraform|ansible)\b',
                r'\b(jira|confluence|figma|sketch|postman|swagger)\b'
            ],
            'soft_skill': [
                r'\b(leadership|communication|teamwork|problem.solving|project.management)\b',
                r'\b(analytical|critical.thinking|time.management|agile|scrum)\b'
            ],
            'certification': [
                r'\b(aws certified|azure certified|google cloud|gcp|pmp|cissp|pmp|scrum master)\b'
            ]
        }
    
    def _build_company_patterns(self) -> List:
        """Build patterns for company name extraction."""
        return [
            r'(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Group|Technologies?|Solutions?|Software)',
            r'(?:Google|Amazon|Microsoft|Apple|Facebook|Meta|Netflix|Uber|LinkedIn)',
            r'(?:IBM|Oracle|Salesforce|Adobe|Intel|NVIDIA|AMD|Qualcomm)'
        ]
    
    def _build_degree_patterns(self) -> Dict:
        """Build patterns for education degree extraction."""
        return {
            'bachelor': r"\b(b\.?s\.?|b\.?a\.?|bachelor|undergraduate)\b",
            'master': r"\b(m\.?s\.?|m\.?a\.?|master|mba|m\.eng\.)\b",
            'phd': r"\b(ph\.?d\.?|doctorate|doctoral|postdoctoral)\b",
            'associate': r"\b(a\.?s\.?|a\.?a\.?|associate)\b",
            'certificate': r"\b(certificate|certification|diploma)\b"
        }
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all entities from resume text.
        
        Args:
            text: Raw resume text
            
        Returns:
            Dictionary containing all extracted entities
        """
        doc = self.nlp(text)
        
        # Initialize result dictionary
        result = {
            'skills': {
                'programming_languages': [],
                'frameworks': [],
                'databases': [],
                'tools': [],
                'soft_skills': [],
                'certifications': []
            },
            'experience': [],
            'education': [],
            'personal_info': {},
            'raw_entities': []
        }
        
        # Extract using spaCy NER
        for ent in doc.ents:
            result['raw_entities'].append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # Extract using custom patterns
        if self.enable_custom_rules:
            self._extract_skills(text, result)
            self._extract_education(text, result)
            self._extract_experience_patterns(text, result)
        
        # Clean and deduplicate
        result = self._clean_results(result)
        
        return result
    
    def _extract_skills(self, text: str, result: Dict):
        """Extract skills using regex patterns."""
        text_lower = text.lower()
        
        for category, patterns in self.skill_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if category == 'programming_language':
                        result['skills']['programming_languages'].append(match.lower())
                    elif category == 'framework':
                        result['skills']['frameworks'].append(match.lower())
                    elif category == 'database':
                        result['skills']['databases'].append(match.lower())
                    elif category == 'tool':
                        result['skills']['tools'].append(match.lower())
                    elif category == 'soft_skill':
                        result['skills']['soft_skills'].append(match.lower())
                    elif category == 'certification':
                        result['skills']['certifications'].append(match.lower())
    
    def _extract_education(self, text: str, result: Dict):
        """Extract education information."""
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Extract degree level
            for degree_type, pattern in self.degree_patterns.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    education_entry = {'type': degree_type, 'raw': sentence}
                    
                    # Extract field of study
                    field_match = re.search(
                        r"(?:in|of|studies?)\s+([A-Za-z\s]+?)(?:\.|,|$)",
                        sentence
                    )
                    if field_match:
                        education_entry['field'] = field_match.group(1).strip()
                    
                    # Extract university
                    uni_match = re.search(
                        r"(?:university|college|institute|school)\s+of\s+([A-Za-z\s]+)",
                        sentence, re.IGNORECASE
                    )
                    if not uni_match:
                        uni_match = re.search(
                            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:University|College)",
                            sentence
                        )
                    if uni_match:
                        education_entry['institution'] = uni_match.group(1).strip()
                    
                    result['education'].append(education_entry)
                    break
    
    def _extract_experience_patterns(self, text: str, result: Dict):
        """Extract work experience patterns."""
        # Find job titles
        job_title_patterns = [
            r'\b(senior|junior|lead|principal|staff|chief|head|director|manager|engineer|developer|analyst|designer|consultant|specialist)\s+[A-Za-z\s]+',
            r'\b(cto|ceo|cfo|vp|vp of|head of)\b'
        ]
        
        for pattern in job_title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 5:
                    result['experience'].append({
                        'job_title': match.strip(),
                        'raw': match.strip()
                    })
    
    def _clean_results(self, result: Dict) -> Dict:
        """Clean and deduplicate extracted results."""
        for category in result['skills']:
            unique_skills = list(set(result['skills'][category]))
            result['skills'][category] = sorted(unique_skills)
        
        # Remove duplicate education entries
        seen = set()
        unique_education = []
        for edu in result['education']:
            key = edu.get('type', '') + edu.get('field', '')
            if key not in seen:
                seen.add(key)
                unique_education.append(edu)
        result['education'] = unique_education
        
        return result
    
    def extract_skills_list(self, text: str) -> List[str]:
        """Extract all skills as a flat list."""
        result = self.extract(text)
        all_skills = []
        for category, skills in result['skills'].items():
            all_skills.extend(skills)
        return list(set(all_skills))
    
    def get_skill_overlap(
        self,
        resume_skills: List[str],
        job_skills: List[str]
    ) -> float:
        """
        Calculate Jaccard similarity between resume and job skills.
        
        Args:
            resume_skills: List of skills from resume
            job_skills: List of required skills from job description
            
        Returns:
            Jaccard similarity score (0-1)
        """
        resume_set = set(skill.lower() for skill in resume_skills)
        job_set = set(skill.lower() for skill in job_skills)
        
        intersection = len(resume_set & job_set)
        union = len(resume_set | job_set)
        
        if union == 0:
            return 0.0
        
        return intersection / union


class HardNegativeMiner:
    """
    Mining module for identifying hard negative examples.
    Used to improve Model 1 training with challenging pairs.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_negatives_per_positive: int = 5
    ):
        """
        Initialize the hard negative miner.
        
        Args:
            similarity_threshold: Minimum similarity to consider as hard negative
            max_negatives_per_positive: Maximum negatives to mine per positive
        """
        self.similarity_threshold = similarity_threshold
        self.max_negatives_per_positive = max_negatives_per_positive
        
        # TF-IDF vectorizer for similarity computation
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def mine_hard_negatives(
        self,
        positives: List[Dict],
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        Mine hard negative examples from candidate pool.
        
        Args:
            positives: List of positive (good match) examples
            candidates: List of candidate resume-job pairs
            
        Returns:
            List of hard negative examples
        """
        # Compute TF-IDF vectors for all texts
        all_texts = [p['resume_text'] for p in positives] + \
                   [c['resume_text'] for c in candidates]
        
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        positives_vectors = tfidf_matrix[:len(positives)]
        candidates_vectors = tfidf_matrix[len(positives):]
        
        # Compute similarities
        similarities = cosine_similarity(positives_vectors, candidates_vectors)
        
        hard_negatives = []
        
        for pos_idx, pos in enumerate(positives):
            # Get similarities for this positive
            pos_similarities = similarities[pos_idx]
            
            # Find top similar candidates (but not too similar)
            sorted_indices = np.argsort(pos_similarities)[::-1]
            
            negatives_mined = 0
            for cand_idx in sorted_indices:
                if negatives_mined >= self.max_negatives_per_positive:
                    break
                
                sim_score = pos_similarities[cand_idx]
                candidate = candidates[cand_idx]
                
                # Check if it's a hard negative (high similarity but negative label)
                if (sim_score >= self.similarity_threshold and 
                    candidate.get('label') == 'bad_fit'):
                    
                    hard_negatives.append({
                        'resume_text': candidate['resume_text'],
                        'job_description': pos['job_description'],
                        'similarity': float(sim_score),
                        'label': 'hard_negative',
                        'reason': 'high_similarity_bad_label'
                    })
                    negatives_mined += 1
        
        return hard_negatives
    
    def compute_keyword_overlap(
        self,
        text1: str,
        text2: str
    ) -> Dict[str, float]:
        """
        Compute keyword-based overlap between two texts.
        
        Args:
            text1: First text (resume)
            text2: Second text (job description)
            
        Returns:
            Dictionary with overlap metrics
        """
        # Extract keywords (simple approach)
        words1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text1.lower()))
        words2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text2.lower()))
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'have', 'been', 'were', 'they', 'their', 'with', 'this'}
        words1 -= stopwords
        words2 -= stopwords
        
        # Calculate overlaps
        intersection = words1 & words2
        union = words1 | words2
        
        jaccard = len(intersection) / len(union) if union else 0
        precision = len(intersection) / len(words1) if words1 else 0
        recall = len(intersection) / len(words2) if words2 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        
        return {
            'jaccard': jaccard,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'common_keywords': list(intersection)
        }
    
    def select_top_k(
        self,
        candidates: List[Dict],
        job_description: str,
        k: int,
        use_extractor: ResumeExtractor = None
    ) -> List[Dict]:
        """
        Select top K candidates based on skill overlap.
        
        Args:
            candidates: List of candidate resumes
            job_description: Job description text
            k: Number of candidates to select
            use_extractor: Optional extractor for skill-based filtering
            
        Returns:
            Top K candidates sorted by relevance
        """
        # Compute basic keyword overlap
        for candidate in candidates:
            overlap = self.compute_keyword_overlap(
                candidate['resume_text'],
                job_description
            )
            candidate['keyword_overlap'] = overlap['jaccard']
            candidate['keyword_f1'] = overlap['f1']
            candidate['common_keywords'] = overlap['common_keywords']
        
        # If extractor provided, also compute skill overlap
        if use_extractor:
            job_skills = use_extractor.extract_skills_list(job_description)
            
            for candidate in candidates:
                resume_skills = use_extractor.extract_skills_list(
                    candidate['resume_text']
                )
                skill_overlap = use_extractor.get_skill_overlap(
                    resume_skills, job_skills
                )
                candidate['skill_overlap'] = skill_overlap
        
        # Sort by combined score
        for candidate in candidates:
            if 'skill_overlap' in candidate:
                # Combine keyword and skill overlap
                candidate['combined_score'] = (
                    0.4 * candidate['keyword_overlap'] +
                    0.6 * candidate['skill_overlap']
                )
            else:
                candidate['combined_score'] = candidate['keyword_overlap']
        
        # Sort and select top K
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return sorted_candidates[:k]


class ExtractionTrainer:
    """Training utilities for entity extraction model."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            # (Implementation depends on specific extraction model)
            loss = self._compute_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'avg_loss': total_loss / num_batches}
    
    def _compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for extraction model."""
        # Placeholder - implementation depends on model architecture
        return torch.tensor(0.0, device=self.device)
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate extraction model."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        for batch in dataloader:
            # Get predictions
            predictions = self._get_predictions(batch)
            all_predictions.extend(predictions)
            all_targets.extend(batch['labels'])
        
        # Calculate precision, recall, F1
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _get_predictions(self, batch) -> List:
        """Get predictions from model."""
        # Placeholder - implementation depends on model architecture
        return []
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)
