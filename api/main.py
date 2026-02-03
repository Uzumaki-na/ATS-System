"""
FastAPI Application for TriadRank ATS
REST API for resume ranking and scoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from contextlib import asynccontextmanager
import logging
import uuid
from datetime import datetime

from pipeline.inference import TriadRankPipeline, CandidateResult
from config import config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global pipeline instance
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global pipeline
    logger.info("Starting TriadRank ATS API...")
    
    # Initialize pipeline on startup
    try:
        pipeline = TriadRankPipeline(
            model1_path=config['models']['cross_encoder'].get('save_path'),
            model2_path=config['models']['category_encoder'].get('save_path'),
            model3_spacy_model=config['models']['extractor'].get('spacy_model', 'en_core_web_sm'),
            device=config['inference']['device'],
            top_k=config['inference']['top_k'],
            category_penalty=config['penalties']['category_mismatch']
        )
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = None
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down TriadRank ATS API...")


# Create FastAPI app
app = FastAPI(
    title="TriadRank ATS API",
    description="3-Tier Resume Scoring and Ranking System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api'].get('cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Pydantic Models for Request/Response
# ============================================

class JobCategory(str, Enum):
    """Supported job categories."""
    ACCOUNTANT = "ACCOUNTANT"
    ADVOCATE = "ADVOCATE"
    AGRICULTURE = "AGRICULTURE"
    APPAREL = "APPAREL"
    ARTS = "ARTS"
    AUTOMOBILE = "AUTOMOBILE"
    AVIATION = "AVIATION"
    BANKING = "BANKING"
    BPO = "BPO"
    BUSINESS_DEVELOPMENT = "BUSINESS-DEVELOPMENT"
    CHEF = "CHEF"
    CONSTRUCTION = "CONSTRUCTION"
    CONSULTANT = "CONSULTANT"
    DESIGNER = "DESIGNER"
    DIGITAL_MEDIA = "DIGITAL-MEDIA"
    ENGINEERING = "ENGINEERING"
    FINANCE = "FINANCE"
    FITNESS = "FITNESS"
    HEALTHCARE = "HEALTHCARE"
    HUMAN_RESOURCES = "HUMAN-RESOURCES"
    INFORMATION_TECHNOLOGY = "INFORMATION-TECHNOLOGY"
    PUBLIC_RELATIONS = "PUBLIC-RELATIONS"
    SALES = "SALES"
    TEACHER = "TEACHER"


class CandidateInput(BaseModel):
    """Input candidate for ranking."""
    id: str = Field(..., description="Unique candidate identifier")
    text: str = Field(..., description="Resume text content")


class RankRequest(BaseModel):
    """Request body for ranking candidates."""
    job_description: str = Field(..., description="Job description text")
    job_category: JobCategory = Field(..., description="Target job category")
    candidates: List[CandidateInput] = Field(..., description="List of candidates to rank")
    top_k: Optional[int] = Field(None, description="Number of top candidates to return")


class LabelProbability(BaseModel):
    """Label probability distribution."""
    good_fit: float
    potential_fit: float
    bad_fit: float


class CategoryResult(BaseModel):
    """Category validation result."""
    predicted: str
    match: bool
    confidence: float


class CandidateOutput(BaseModel):
    """Output candidate ranking result."""
    rank: int
    candidate_id: str
    final_score: float
    raw_score: float
    label: str
    label_probabilities: LabelProbability
    category: CategoryResult
    skill_overlap: float
    keyword_overlap: float
    processing_time: float
    extracted_skills: List[str]


class RankResponse(BaseModel):
    """Response body for ranking operation."""
    job_id: str
    job_category: str
    total_candidates: int
    returned_candidates: int
    processing_time_seconds: float
    results: List[CandidateOutput]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pipeline_ready: bool
    models_loaded: bool
    timestamp: str


class CategoryListResponse(BaseModel):
    """List of supported categories."""
    categories: List[str]
    count: int


# ============================================
# API Endpoints
# ============================================

@app.get("/", summary="API Root")
async def root():
    """API root endpoint."""
    return {
        "name": "TriadRank ATS API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """
    Check the health status of the API and pipeline.
    """
    global pipeline
    
    pipeline_ready = pipeline is not None
    models_loaded = False
    
    if pipeline_ready:
        models_loaded = (
            hasattr(pipeline, 'model1') and 
            hasattr(pipeline, 'model2') and 
            hasattr(pipeline, 'extractor')
        )
    
    status = "healthy" if (pipeline_ready and models_loaded) else "degraded"
    
    return HealthResponse(
        status=status,
        pipeline_ready=pipeline_ready,
        models_loaded=models_loaded,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/categories", response_model=CategoryListResponse, summary="List Categories")
async def list_categories():
    """
    List all supported job categories.
    """
    categories = list(config['categories']['id_to_name'].values())
    
    return CategoryListResponse(
        categories=categories,
        count=len(categories)
    )


@app.post("/rank", response_model=RankResponse, summary="Rank Candidates")
async def rank_candidates(request: RankRequest):
    """
    Rank candidates against a job description.
    
    This endpoint processes resumes through the 3-tier system:
    1. Tier 3: Entity extraction and skill-based filtering
    2. Tier 2: Category validation with penalty logic
    3. Tier 3: Deep cross-encoding for final scoring
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Please check /health endpoint."
        )
    
    import time
    start_time = time.time()
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Convert candidates to pipeline format
        candidates_data = [
            {"id": c.id, "text": c.text}
            for c in request.candidates
        ]
        
        # Override top_k if provided
        top_k = request.top_k or config['inference']['top_k']
        
        # Run pipeline
        results = pipeline.rank_candidates(
            job_description=request.job_description,
            job_category=request.job_category.value,
            candidates=candidates_data,
            top_k=top_k
        )
        
        # Convert to output format
        output_results = []
        for rank, result in enumerate(results, 1):
            output = CandidateOutput(
                rank=rank,
                candidate_id=result.candidate_id,
                final_score=round(result.final_score, 4),
                raw_score=round(result.raw_score, 4),
                label=result.label,
                label_probabilities=LabelProbability(
                    good_fit=result.label_probabilities.get('Good Fit', 0),
                    potential_fit=result.label_probabilities.get('Potential Fit', 0),
                    bad_fit=result.label_probabilities.get('Bad Fit', 0)
                ),
                category=CategoryResult(
                    predicted=result.category_predicted,
                    match=result.category_match,
                    confidence=round(result.category_confidence, 4)
                ),
                skill_overlap=round(result.skill_overlap, 4),
                keyword_overlap=round(result.keyword_overlap, 4),
                processing_time=round(result.processing_time, 4),
                extracted_skills=result.metadata.get('extracted_skills', []) if result.metadata else []
            )
            output_results.append(output)
        
        processing_time = time.time() - start_time
        
        return RankResponse(
            job_id=job_id,
            job_category=request.job_category.value,
            total_candidates=len(request.candidates),
            returned_candidates=len(results),
            processing_time_seconds=round(processing_time, 4),
            results=output_results
        )
    
    except Exception as e:
        logger.error(f"Error processing ranking request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/rank/single", summary="Rank Single Candidate")
async def rank_single(
    candidate_text: str,
    job_description: str,
    job_category: JobCategory
):
    """
    Rank a single candidate against a job description.
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized."
        )
    
    try:
        result = pipeline.rank_single(
            resume_text=candidate_text,
            job_description=job_description,
            job_category=job_category.value
        )
        
        return {
            "candidate_id": result.candidate_id,
            "final_score": round(result.final_score, 4),
            "raw_score": round(result.raw_score, 4),
            "label": result.label,
            "category_match": result.category_match,
            "category_predicted": result.category_predicted,
            "skill_overlap": round(result.skill_overlap, 4),
            "keyword_overlap": round(result.keyword_overlap, 4)
        }
    
    except Exception as e:
        logger.error(f"Error in single ranking: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/extract", summary="Extract Entities from Resume")
async def extract_entities(resume_text: str):
    """
    Extract entities (skills, experience, education) from a resume.
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized."
        )
    
    try:
        extraction = pipeline.extractor.extract(resume_text)
        return extraction
    
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/categories/{category_id}/name", summary="Get Category Name")
async def get_category_name(category_id: int):
    """
    Get category name by ID.
    """
    id_to_name = config['categories']['id_to_name']
    
    if category_id not in id_to_name:
        raise HTTPException(
            status_code=404,
            detail=f"Category ID {category_id} not found."
        )
    
    return {
        "id": category_id,
        "name": id_to_name[category_id]
    }


# ============================================
# Error Handlers
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."}
    )


# ============================================
# Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api'].get('debug', False)
    )
