"""
Pipeline package for TriadRank ATS.
Contains the main inference pipeline.
"""

from pipeline.inference import (
    TriadRankPipeline,
    CandidateResult,
)

__all__ = [
    'TriadRankPipeline',
    'CandidateResult',
]
