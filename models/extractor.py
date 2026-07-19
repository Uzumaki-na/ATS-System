"""
Model 3: Entity Extractor for Resume Parsing
Extracts skills, experience, education, and other entities from resumes.
Also handles hard negative mining for training.

Skill extraction uses a 31K+ Aho-Corasick trie built from SkillNer
SKILL_DB + curated aliases.  Regex / SkillNer / HF NER remain as
fallbacks for backwards compatibility.
"""

import torch
import torch.nn as nn
import spacy
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import config
from models.skill_trie import SkillTrie

logger = logging.getLogger(__name__)


class ResumeExtractor:
    """
    Entity extraction module for parsing resumes.
    Uses spaCy NER with custom rules for resume-specific entities.
    Optionally uses SkillNer (EMSI skills database) or a HuggingFace
    token-classification model for skill extraction.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_custom_rules: bool = True,
        skill_extractor_mode: str = "trie",
        skill_ner_model: str = None,
        skill_db_path: Optional[str] = None,
    ):
        """
        Initialize the extractor with spaCy model.

        Args:
            spacy_model: spaCy model name to load
            enable_custom_rules: Whether to enable custom extraction rules
            skill_extractor_mode: "trie" | "regex" | "skillner" | "hf_ner".
                "trie" = Aho-Corasick trie backed by 31K SkillNer DB + curated
                         aliases (default).  Instant, scalable, no spaCy overhead.
                "regex" = pattern-based (legacy).
                "skillner" = EMSI skills database via PhraseMatcher (31K skills).
                "hf_ner" = HuggingFace token-classification model.
            skill_ner_model: HuggingFace model name for token-classification
                            (used when mode="hf_ner").
            skill_db_path: Path to skill_db.json for "trie" mode.
                           Defaults to <project_root>/data/skill_db.json.
        """
        self.nlp = spacy.load(spacy_model)
        self.enable_custom_rules = enable_custom_rules
        self.skill_extractor_mode = skill_extractor_mode
        self.skill_ner_model = skill_ner_model

        # Custom skill patterns (fallback when regex mode)
        self.skill_patterns = self._build_skill_patterns()

        # Company name patterns
        self.company_patterns = self._build_company_patterns()

        # Degree patterns
        self.degree_patterns = self._build_degree_patterns()

        # SkillNer extractor (lazy-loaded)
        self._skillner_extractor = None

        # HuggingFace NER pipeline (lazy-loaded)
        self._ner_pipeline = None

        # Aho-Corasick trie (default mode) — 31K+ skills, instant match
        self._skill_trie = None
        if skill_extractor_mode == "trie":
            if skill_db_path is None:
                # Default path: <project_root>/data/skill_db.json
                _root = Path(__file__).resolve().parent.parent
                skill_db_path = str(_root / "data" / "skill_db.json")
            if Path(skill_db_path).exists():
                self._skill_trie = SkillTrie(skill_db_path)
                logger.info(
                    "SkillTrie loaded: %d skills, %d aliases",
                    self._skill_trie.skill_count,
                    len(self._skill_trie._canonical),
                )
            else:
                logger.warning(
                    "Skill database not found at '%s' — "
                    "run scripts/build_skill_db.py first. "
                    "Falling back to regex mode.",
                    skill_db_path,
                )
                self.skill_extractor_mode = "regex"

    def _load_skillner(self):
        """Lazy-load the SkillNer extractor (EMSI skills database)."""
        if self._skillner_extractor is not None:
            return
        try:
            from spacy.matcher import PhraseMatcher
            from skillNer.general_params import SKILL_DB
            from skillNer.skill_extractor_class import SkillExtractor as _SE

            self._skillner_extractor = _SE(self.nlp, SKILL_DB, PhraseMatcher)
            logger.info("SkillNer loaded (%d skills in DB)", len(SKILL_DB))
        except Exception:
            logger.warning(
                "SkillNer not available, falling back to regex patterns. "
                "Install: pip install skillNer ipython",
                exc_info=True,
            )
            self.skill_extractor_mode = "regex"

    def _load_hf_ner(self):
        """Lazy-load the HuggingFace token-classification pipeline."""
        if self._ner_pipeline is not None:
            return
        if not self.skill_ner_model:
            return
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
        tokenizer = AutoTokenizer.from_pretrained(self.skill_ner_model)
        model = AutoModelForTokenClassification.from_pretrained(self.skill_ner_model)
        self._ner_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
    
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
            'ner_skills': [],  # flat list from SkillNer/HF NER when available
            'experience': [],
            'experience_years': 0,  # extracted from explicit mentions and date ranges
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

        # Extract skills — dispatch by configured mode
        if self.skill_extractor_mode == "trie" and self._skill_trie is not None:
            self._extract_skills_trie(text, result)
        elif self.skill_extractor_mode == "skillner":
            if self._skillner_extractor is None:
                self._load_skillner()
            if self._skillner_extractor is not None:
                self._extract_skills_skillner(text, result)
            elif self.enable_custom_rules:
                self._extract_skills(text, result)
        elif self.skill_extractor_mode == "hf_ner":
            self._load_hf_ner()
            self._extract_skills_ner(text, result)
        elif self.enable_custom_rules:
            self._extract_skills(text, result)

        # Extract education and experience (always keep these)
        if self.enable_custom_rules:
            self._extract_education(text, result)
            self._extract_experience_patterns(text, result)

        # Clean and deduplicate
        result = self._clean_results(result)

        return result
    
    def _extract_skills_trie(self, text: str, result: Dict):
        """Extract skills using the Aho-Corasick trie (31K+ skills)."""
        matches = self._skill_trie.match_categorized(text)
        for cat, skills in matches.items():
            # Map trie categories to result dict keys
            if cat in ("programming_language",):
                result["skills"]["programming_languages"].extend(skills)
            elif cat in ("framework",):
                result["skills"]["frameworks"].extend(skills)
            elif cat in ("database",):
                result["skills"]["databases"].extend(skills)
            elif cat in ("tool", "devops"):
                result["skills"]["tools"].extend(skills)
            elif cat in ("soft_skill",):
                result["skills"]["soft_skills"].extend(skills)
            elif cat in ("certification",):
                result["skills"]["certifications"].extend(skills)
            else:
                result["ner_skills"].extend(skills)

    def _extract_skills_skillner(self, text: str, result: Dict):
        """Extract skills using SkillNer's EMSI skills database."""
        if self._skillner_extractor is None:
            return
        try:
            annotations = self._skillner_extractor.annotate(text)
            results = annotations.get("results", {})
            seen = set()
            for bucket in ("full_matches", "ngram_scored", "abv_matches"):
                for m in results.get(bucket, []):
                    skill = m.get("doc_node_value", "").strip()
                    if skill and skill.lower() not in seen:
                        seen.add(skill.lower())
                        result["ner_skills"].append(skill)
        except Exception:
            logger.warning("SkillNer extraction failed, falling back to regex", exc_info=True)
            self._extract_skills(text, result)

    def _extract_skills_ner(self, text: str, result: Dict):
        """Extract skills using HuggingFace token-classification NER model."""
        if self._ner_pipeline is None:
            return
        try:
            predictions = self._ner_pipeline(text)
            seen = set()
            for pred in predictions:
                skill = pred.get("word", "").strip()
                # The aggregation_strategy="simple" handles B/I merging,
                # but tokens may include ## subword fragments (WordPiece).
                # Clean those up.
                skill = skill.replace(" ##", "").replace("##", "")
                if skill and skill.lower() not in seen:
                    seen.add(skill.lower())
                    result["ner_skills"].append(skill)
        except Exception:
            logger.warning("NER skill extraction failed, falling back to regex", exc_info=True)
            self._extract_skills(text, result)

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
        """Extract work experience patterns — titles, years, date ranges."""
        # Job title patterns
        job_title_patterns = [
            r'\b(senior|junior|lead|principal|staff|chief|head|director|manager|engineer|developer|analyst|designer|consultant|specialist|architect|scientist|coordinator|associate|intern)\s+[A-Za-z\s]+',
            r'\b(cto|ceo|cfo|coo|vp|vp of|head of|director of|manager of)\b'
        ]

        for pattern in job_title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 5:
                    result['experience'].append({
                        'job_title': match.strip(),
                        'raw': match.strip()
                    })

        # Years of experience — explicit mention patterns
        years_total = 0.0
        years_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
            r'experience\s+(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)',
            r'(?:over|>)\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:in|with|of)',
        ]
        for pattern in years_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = float(m.group(1))
                years_total = max(years_total, val)

        # Date range patterns — compute years from work history spans.
        # Use two strategies: month-inclusive (preferred) then simple year-only.
        # Track matched positions to avoid double-counting.
        date_ranges_found = []
        matched_spans = []  # (start, end) char positions already counted

        def _span_overlaps(s, e, existing):
            return any(not (e <= es or s >= ee) for es, ee in existing)

        # Strategy 1: "Jun 2019 – Present" or "June 2019 - Present"
        month_range_pat = (
            r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*'
            r'(?:\s+\d{4})?)\s*[-–—to]+\s*'
            r'(present|current|now|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*(?:\s+\d{4})?\b)'
        )
        for m in re.finditer(month_range_pat, text, re.IGNORECASE):
            start_str, end_str = m.group(1), m.group(2)
            if _span_overlaps(m.start(), m.end(), matched_spans):
                continue
            is_present = end_str.lower() in ('present', 'current', 'now')
            start_parts = re.findall(r'\b(\d{4})\b', start_str)
            end_parts = re.findall(r'\b(\d{4})\b', end_str) if not is_present else []
            if start_parts:
                start_year = int(start_parts[0])
                end_year = 2026 if is_present else (int(end_parts[0]) if end_parts else start_year + 1)
                date_ranges_found.append(max(1, end_year - start_year))
                matched_spans.append((m.start(), m.end()))

        # Strategy 2: "2018 - 2023" (only if not already counted)
        for m in re.finditer(r'(\b\d{4})\s*[-–—to]+\s*(\d{4}|present|current|now)', text, re.IGNORECASE):
            if _span_overlaps(m.start(), m.end(), matched_spans):
                continue
            start_str, end_str = m.group(1), m.group(2)
            is_present = end_str.lower() in ('present', 'current', 'now')
            start_year = int(start_str)
            end_year = 2026 if is_present else int(end_str)
            date_ranges_found.append(max(1, end_year - start_year))
            matched_spans.append((m.start(), m.end()))

        if date_ranges_found:
            years_total = max(years_total, sum(date_ranges_found))

        # Also try to infer from number of positions listed
        if years_total == 0:
            position_markers = len(re.findall(
                r'\b(?:experience|employment|work history|professional)\b',
                text, re.IGNORECASE
            ))
            if position_markers >= 2:
                years_total = max(1.0, position_markers * 1.5)

        if years_total > 0:
            result['experience_years'] = years_total
    
    def _clean_results(self, result: Dict) -> Dict:
        """Clean and deduplicate extracted results."""
        for category in result['skills']:
            unique_skills = list(set(result['skills'][category]))
            result['skills'][category] = sorted(unique_skills)

        # Deduplicate NER skills
        seen = set()
        unique_ner = []
        for s in result.get("ner_skills", []):
            key = s.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique_ner.append(s)
        result["ner_skills"] = unique_ner

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
        """Extract all skills as a flat list.
        Prefers SkillNer / NER results when available, falls back to regex.
        """
        result = self.extract(text)
        if result.get("ner_skills"):
            return list(result["ner_skills"])
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
