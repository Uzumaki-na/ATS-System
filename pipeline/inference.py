"""
Main Pipeline: TriadRank Inference Engine
Chains Model 3 (Extraction) → Model 2 (Category) → Model 1 (Scoring)
"""

import torch
import logging
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
from pathlib import Path

from models.cross_encoder import MultiHeadCrossEncoder
from models.category_encoder import CategoryEncoder
from models.extractor import ResumeExtractor, HardNegativeMiner
from data.loaders import load_inference_batch, clean_resume_text, validate_category
from config import config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_artifact_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None

    p = Path(path)
    if p.is_absolute():
        return str(p)

    project_root = Path(__file__).resolve().parent.parent
    return str((project_root / p).resolve())


@dataclass
class CandidateResult:
    """Result for a single candidate after full pipeline processing."""
    candidate_id: str
    final_score: float
    raw_score: float
    label: str
    label_probabilities: Dict[str, float]
    category_match: bool
    category_predicted: str
    category_confidence: float
    skill_overlap: float
    keyword_overlap: float
    ner_skills: List[str] = None       # skills from SkillNer / NER extraction
    experience_years: float = 0        # extracted years of experience
    education: List = None             # extracted education entries
    processing_time: float = 0
    metadata: Dict = None


class TriadRankPipeline:
    """
    Main inference pipeline for ATS scoring and ranking.
    
    Flow:
    1. Model 3: Extract entities and select top-K candidates
    2. Model 2: Check category match and apply penalty
    3. Model 1: Score candidates with cross-encoder
    4. Combine scores and return ranked list
    """
    
    def __init__(
        self,
        model1_path: str = None,
        model2_path: str = None,
        model3_spacy_model: str = "en_core_web_sm",
        device: str = "cuda",
        top_k: int = 50,
        category_penalty: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize the pipeline with all three models.
        
        Args:
            model1_path: Path to Cross-Encoder weights
            model2_path: Path to Category Encoder weights
            model3_spacy_model: spaCy model for extraction
            device: Device to run models on ('cuda' or 'cpu')
            top_k: Number of candidates to pass from Tier 3 to Tier 2
            category_penalty: Penalty factor for category mismatch
            verbose: Whether to log progress
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
        self.top_k = top_k
        self.category_penalty = category_penalty
        self.category_similarity = config.get('penalties', {}).get('category_similarity', {})
        self.verbose = verbose

        tier3_cfg = (config.get('inference', {}) or {}).get('tier3', {}) or {}
        self.tier3_use_skill_overlap = bool(tier3_cfg.get('use_skill_overlap', True))
        self.tier3_keyword_weight = float(tier3_cfg.get('keyword_weight', 0.4))
        self.tier3_skill_weight = float(tier3_cfg.get('skill_weight', 0.6))
        self.tier3_enable_custom_rules = bool(tier3_cfg.get('enable_custom_rules', True))

        # Final-score blending. Cross-encoder alone collapses to ~0.46 for every
        # pair (training data has no negatives), so raw_score * penalty_factor
        # yields no real ranking signal. The weighted mode blends content overlap
        # (which actually depends on resume vs JD text) with the category gate.
        scoring_cfg = (config.get('inference', {}) or {}).get('scoring', {}) or {}
        self.scoring_mode = scoring_cfg.get('mode', 'weighted')
        _w = scoring_cfg.get('weights', {}) or {}
        self.w_content = float(_w.get('content_overlap', 0.50))
        self.w_category = float(_w.get('category', 0.30))
        self.w_cross = float(_w.get('cross_encoder', 0.20))
        
        # Get category mappings from config (fallback; may be overridden by checkpoint)
        self.id_to_name = config['categories']['id_to_name']
        self.name_to_id = config['categories']['name_to_id']

        # Cache tokenizers once per pipeline lifetime (I1)
        from transformers import AutoTokenizer
        self._cat_tokenizer = AutoTokenizer.from_pretrained(
            config['models']['category_encoder']['pretrained_model'],
            token=False,
        )
        self._cross_tokenizer = AutoTokenizer.from_pretrained(
            config['models']['cross_encoder']['pretrained_model'],
            token=False,
        )

        # Initialize models
        self._init_models(model1_path, model2_path, model3_spacy_model)

        logger.info(f"Pipeline initialized on device: {device}")

    def _get_category_penalty(self, pred_category: str, target_category: str) -> float:
        """Get penalty factor based on category similarity.

        Args:
            pred_category: Predicted category name
            target_category: Target job category name

        Returns:
            Penalty factor (1.0 = no penalty, lower = harsher penalty)
        """
        if pred_category == target_category:
            return 1.0
        # Check similarity matrix (defined once per pair in config)
        pair = f"{pred_category},{target_category}"
        rev = f"{target_category},{pred_category}"
        similarity = self.category_similarity.get(pair) or self.category_similarity.get(rev)
        if similarity is not None:
            return similarity
        return self.category_penalty
    
    def _init_models(
        self,
        model1_path: str,
        model2_path: str,
        spacy_model: str
    ):
        """Initialize and load all three models."""
        
        # Model 3: Entity Extractor (no training needed, uses spaCy)
        logger.info("Initializing Model 3: Entity Extractor...")
        resolved_spacy_model = _resolve_artifact_path(spacy_model)
        if resolved_spacy_model and Path(resolved_spacy_model).exists():
            spacy_model_to_load = resolved_spacy_model
        else:
            spacy_model_to_load = spacy_model
        skill_mode = config.get("inference", {}).get("tier3", {}).get("skill_extractor_mode", "regex")
        self.extractor = ResumeExtractor(
            spacy_model=spacy_model_to_load,
            enable_custom_rules=self.tier3_enable_custom_rules,
            skill_extractor_mode=skill_mode
        )
        self.miner = HardNegativeMiner()
        logger.info("Model 3 initialized successfully")
        
        # Model 2: Category Encoder
        logger.info("Initializing Model 2: Category Encoder...")
        resolved_model2_path = _resolve_artifact_path(model2_path)
        if resolved_model2_path and Path(resolved_model2_path).exists():
            self.model2 = CategoryEncoder.load(resolved_model2_path, device=self.device)
            # B3: prefer the checkpoint's own label space over config['categories']
            # (which may have a different class count, e.g. 24 vs 25).
            if getattr(self.model2, 'id_to_name', None) is not None:
                self.id_to_name = self.model2.id_to_name
                self.name_to_id = self.model2.category_to_idx
                logger.info(
                    f"B3: Using checkpoint label mapping ({len(self.id_to_name)} classes)"
                )
        else:
            msg = (
                "Model 2 checkpoint not found at '%s' — falling back to random "
                "UNTRAINED weights. All category predictions will be noise."
            )
            logger.warning(msg, resolved_model2_path)
            warnings.warn(msg % resolved_model2_path, stacklevel=2)
            self.model2 = CategoryEncoder.from_config(
                config['models']['category_encoder']
            )
        self.model2.to(self.device)
        self.model2.eval()
        logger.info("Model 2 loaded successfully")
        
        # Model 1: Cross-Encoder
        logger.info("Initializing Model 1: Cross-Encoder...")
        resolved_model1_path = _resolve_artifact_path(model1_path)
        if resolved_model1_path and Path(resolved_model1_path).exists():
            self.model1 = MultiHeadCrossEncoder.load(resolved_model1_path, device=self.device)
        else:
            msg = (
                "Model 1 checkpoint not found at '%s' — falling back to random "
                "UNTRAINED weights. All scoring outputs will be noise."
            )
            logger.warning(msg, resolved_model1_path)
            warnings.warn(msg % resolved_model1_path, stacklevel=2)
            self.model1 = MultiHeadCrossEncoder.from_config(
                config['models']['cross_encoder']
            )
        self.model1.to(self.device)
        self.model1.eval()
        logger.info("Model 1 loaded successfully")
    
    def rank_candidates(
        self,
        job_description: str,
        job_category: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None,
        return_details: bool = False
    ) -> List[CandidateResult]:
        """
        Rank candidates against a job description.
        
        Args:
            job_description: The job description text
            job_category: The category name (e.g., "Software Engineering")
            candidates: List of candidate dictionaries with 'id' and 'text'
            top_k: Override for number of candidates to keep
            return_details: Whether to include detailed metadata
            
        Returns:
            List of CandidateResult objects sorted by final_score
        """
        start_time = time.time()
        
        # Get target category ID
        normalized_job_category = validate_category(job_category)
        target_category_id = self.name_to_id.get(normalized_job_category)
        if target_category_id is None:
            raise ValueError(f"Unknown job category: {job_category}")
        
        # Override top_k if provided
        if top_k is not None:
            self.top_k = min(top_k, len(candidates))
        
        logger.info(f"Processing {len(candidates)} candidates...")
        
        # ============================================
        # TIER 3: Extraction & Top-K Selection
        # ============================================
        logger.info("Tier 3: Extracting entities and selecting top-K...")
        tier3_start = time.time()
        
        # Compute overlap scores for all candidates
        candidates_with_overlap = self._tier3_extraction(
            candidates, job_description
        )
        
        # Select top-K candidates
        tier3_candidates = candidates_with_overlap[:self.top_k]
        tier3_time = time.time() - tier3_start
        
        logger.info(f"Tier 3: Selected {len(tier3_candidates)} candidates in {tier3_time:.2f}s")
        
        # ============================================
        # TIER 2: Category Validation
        # ============================================
        logger.info("Tier 2: Validating categories...")
        tier2_start = time.time()
        
        tier2_results = self._tier2_category_check(
            tier3_candidates, target_category_id
        )
        tier2_time = time.time() - tier2_start
        
        logger.info(f"Tier 2: Category validation complete in {tier2_time:.2f}s")
        
        # ============================================
        # TIER 1: Deep Scoring
        # ============================================
        logger.info("Tier 1: Cross-encoding for final scoring...")
        tier1_start = time.time()
        
        tier1_results = self._tier1_scoring(
            tier2_results, job_description
        )
        tier1_time = time.time() - tier1_start
        
        logger.info(f"Tier 1: Scoring complete in {tier1_time:.2f}s")
        
        # ============================================
        # Combine and Rank
        # ============================================
        final_results = self._combine_and_rank(tier1_results)
        
        # Add processing time
        total_time = time.time() - start_time
        for result in final_results:
            result.processing_time = total_time
        
        logger.info(f"Total pipeline time: {total_time:.2f}s")
        
        return final_results
    
    def _tier3_extraction(
        self,
        candidates: List[Dict[str, Any]],
        job_description: str
    ) -> List[Dict]:
        """
        Tier 3: Extract entities and compute skill/keyword overlap.
        
        Args:
            candidates: List of candidate dictionaries
            job_description: Job description text
            
        Returns:
            Candidates with overlap scores added
        """
        results = []

        if self.tier3_use_skill_overlap:
            job_skills = self.extractor.extract_skills_list(job_description)
            keyword_weight = self.tier3_keyword_weight
            skill_weight = self.tier3_skill_weight
            if keyword_weight == 0 and skill_weight == 0:
                keyword_weight = 0.4
                skill_weight = 0.6
        else:
            job_skills = []
            keyword_weight = 1.0
            skill_weight = 0.0
        
        for candidate in candidates:
            resume_text = candidate.get('text', '')
            candidate_id = candidate.get('id', 'unknown')

            # Keyword overlap (cheap, always needed)
            keyword_overlap = self.miner.compute_keyword_overlap(
                resume_text, job_description
            )['jaccard']

            # I3: skip the expensive spaCy extract() when skill overlap is disabled
            if self.tier3_use_skill_overlap:
                extraction = self.extractor.extract(resume_text)
                # Use same logic as extract_skills_list(): prefer ner_skills, fall back to regex
                if extraction.get("ner_skills"):
                    resume_skills = list(extraction["ner_skills"])
                else:
                    resume_skills = []
                    for skills in extraction['skills'].values():
                        resume_skills.extend(skills)
                skill_overlap = self.extractor.get_skill_overlap(
                    resume_skills, job_skills
                )
            else:
                extraction = {}
                resume_skills = []
                skill_overlap = 0.0

            combined_score = keyword_weight * keyword_overlap + skill_weight * skill_overlap

            results.append({
                'id': candidate_id,
                'text': resume_text,
                'skills': resume_skills,
                'extraction': extraction,
                'keyword_overlap': keyword_overlap,
                'skill_overlap': skill_overlap,
                'combined_score': combined_score
            })
        
        # Sort by combined score (descending)
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    def _tier2_category_check(
        self,
        candidates: List[Dict],
        target_category_id: int
    ) -> List[Dict]:
        """
        Tier 2: Check if candidate resumes match job category.
        
        Args:
            candidates: Candidates from Tier 3
            target_category_id: Target job category ID
            
        Returns:
            Candidates with category predictions and penalties
        """
        if not candidates:
            return []

        results = []

        tokenizer = self._cat_tokenizer   # I1: cached once in __init__
        max_seq_length = config['models']['category_encoder'].get('max_seq_length', 256)

        resume_texts = [clean_resume_text(c.get('text', '')) for c in candidates]
        encoding = tokenizer(
            resume_texts,
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            predictions = self.model2.predict(input_ids, attention_mask)

        for idx, candidate in enumerate(candidates):
            pred_category_id = predictions['predicted_class'][idx].item()
            confidence = predictions['confidence'][idx].item()
            pred_category_name = self.id_to_name[pred_category_id]

            is_match = pred_category_id == target_category_id
            if is_match:
                penalty_factor = 1.0
            else:
                target_name = self.id_to_name.get(target_category_id, "")
                penalty_factor = self._get_category_penalty(pred_category_name, target_name)

            probs = predictions['probabilities'][idx]
            candidate['category'] = {
                'predicted_id': pred_category_id,
                'predicted_name': pred_category_name,
                'target_id': target_category_id,
                'is_match': is_match,
                'confidence': confidence,
                'penalty_factor': penalty_factor,
                'probabilities': {
                    self.id_to_name[i]: float(probs[i])
                    for i in range(probs.shape[0])
                }
            }

            results.append(candidate)

        return results
    
    def _tier1_scoring(
        self,
        candidates: List[Dict],
        job_description: str
    ) -> List[Dict]:
        """
        Tier 1: Deep scoring with Cross-Encoder.
        
        Args:
            candidates: Candidates from Tier 2
            job_description: Job description text
            
        Returns:
            Candidates with raw scores and labels
        """
        if not candidates:
            return []

        resume_texts = [c.get('text', '') for c in candidates]
        max_seq_length = config['models']['cross_encoder'].get('max_seq_length', 512)
        base_batch_size = int(config.get('inference', {}).get('batch_size', 32) or 32)
        base_batch_size = max(1, base_batch_size)

        label_map = {0: 'Bad Fit', 1: 'Potential Fit', 2: 'Good Fit'}

        start_idx = 0
        current_batch_size = min(base_batch_size, len(candidates))

        while start_idx < len(candidates):
            bs = min(current_batch_size, len(candidates) - start_idx)

            while True:
                try:
                    batch = load_inference_batch(
                        resumes=resume_texts[start_idx:start_idx + bs],
                        job_description=job_description,
                        tokenizer_name=config['models']['cross_encoder']['pretrained_model'],
                        max_seq_length=max_seq_length,
                        tokenizer=self._cross_tokenizer  # I1: cached once in __init__
                    )

                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch.get('token_type_ids', None)
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(self.device)

                    with torch.no_grad():
                        predictions = self.model1.predict(input_ids, attention_mask, token_type_ids=token_type_ids)

                    for local_idx, candidate in enumerate(candidates[start_idx:start_idx + bs]):
                        raw_score = predictions['score'][local_idx].item()
                        pred_class = predictions['predicted_class'][local_idx].item()
                        probs = predictions['probabilities'][local_idx].cpu().numpy()

                        penalty_factor = candidate['category']['penalty_factor']
                        # Weighted blend uses the content-overlap signal from Tier 3
                        # (skill + keyword Jaccard), which actually varies with the
                        # resume-vs-JD text — unlike raw_score, which the cross-encoder
                        # collapses to ~0.46 for everything. Set scoring.mode="legacy"
                        # in config to restore raw_score * penalty_factor.
                        if self.scoring_mode == 'legacy':
                            final_score = raw_score * penalty_factor
                        else:
                            content = candidate.get('combined_score', 0.0)
                            final_score = (
                                self.w_content * content
                                + self.w_category * penalty_factor
                                + self.w_cross * raw_score
                            )

                        candidate['scoring'] = {
                            'raw_score': raw_score,
                            'final_score': final_score,
                            'predicted_class': pred_class,
                            'label': label_map[pred_class],
                            'label_probabilities': {
                                label_map[i]: float(probs[i]) for i in range(len(probs))
                            },
                            'confidence': float(probs[pred_class])
                        }

                    del batch, input_ids, attention_mask, token_type_ids, predictions
                    break
                except RuntimeError as e:
                    is_cuda_oom = (
                        isinstance(e, RuntimeError)
                        and 'out of memory' in str(e).lower()
                        and self.device.startswith('cuda')
                        and torch.cuda.is_available()
                    )
                    if is_cuda_oom and bs > 1:
                        torch.cuda.empty_cache()
                        bs = max(1, bs // 2)
                        current_batch_size = bs
                        logger.warning(
                            f"CUDA OOM during Tier 1 scoring; retrying with batch_size={bs}"
                        )
                        continue
                    raise

            start_idx += bs

        return candidates
    
    def _combine_and_rank(self, candidates: List[Dict]) -> List[CandidateResult]:
        """
        Combine all results and create final ranked list.
        
        Args:
            candidates: Candidates from Tier 1
            
        Returns:
            Sorted list of CandidateResult objects
        """
        results = []
        
        for candidate in candidates:
            scoring = candidate['scoring']
            category = candidate['category']
            extraction = candidate.get('extraction', {})

            result = CandidateResult(
                candidate_id=candidate['id'],
                final_score=scoring['final_score'],
                raw_score=scoring['raw_score'],
                label=scoring['label'],
                label_probabilities=scoring['label_probabilities'],
                category_match=category['is_match'],
                category_predicted=category['predicted_name'],
                category_confidence=category['confidence'],
                skill_overlap=candidate.get('skill_overlap', 0),
                keyword_overlap=candidate.get('keyword_overlap', 0),
                # Unified skills: prefer ner_skills from extraction, fall back to regex
                ner_skills=(extraction.get('ner_skills')
                           if extraction.get('ner_skills')
                           else candidate.get('skills', [])),
                experience_years=extraction.get('experience_years', 0) or 0,
                education=extraction.get('education', []) or [],
                processing_time=0,
                metadata={
                    'penalty_factor': category['penalty_factor'],
                    'experience_years': extraction.get('experience_years', 0),
                }
            )
            
            results.append(result)
        
        # Sort by final score (descending)
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    def rank_single(
        self,
        resume_text: str,
        job_description: str,
        job_category: str
    ) -> CandidateResult:
        """
        Rank a single resume against a job description.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text
            job_category: Target job category name
            
        Returns:
            CandidateResult for this single candidate
        """
        candidates = [{'id': 'single', 'text': resume_text}]
        results = self.rank_candidates(job_description, job_category, candidates)
        
        return results[0]
