"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, root_validator
from typing import Dict, List, Optional, Any
from enum import Enum


class AlertSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class AlertItem(BaseModel):
    """Individual alert for a dimension below threshold"""

    dimension: str = Field(..., description="Dimension name that triggered alert")
    score: float = Field(..., description="Actual score for this dimension")
    threshold: float = Field(..., description="Threshold that was exceeded")
    severity: AlertSeverity = Field(..., description="Alert severity level")


class DimensionScores(BaseModel):
    """19 perceptual dimension scores"""

    timing_stable_unstable: float = Field(..., ge=0, le=1)
    articulation_short_long: float = Field(..., ge=0, le=1)
    articulation_soft_cushioned_hard_solid: float = Field(..., ge=0, le=1)
    pedal_sparse_dry_saturated_wet: float = Field(..., ge=0, le=1)
    pedal_clean_blurred: float = Field(..., ge=0, le=1)
    timbre_even_colorful: float = Field(..., ge=0, le=1)
    timbre_shallow_rich: float = Field(..., ge=0, le=1)
    timbre_bright_dark: float = Field(..., ge=0, le=1)
    timbre_soft_loud: float = Field(..., ge=0, le=1)
    dynamic_sophisticated_mellow_raw_crude: float = Field(..., ge=0, le=1)
    dynamic_little_range_large_range: float = Field(..., ge=0, le=1)
    music_making_fast_paced_slow_paced: float = Field(..., ge=0, le=1)
    music_making_flat_spacious: float = Field(..., ge=0, le=1)
    music_making_disproportioned_balanced: float = Field(..., ge=0, le=1)
    music_making_pure_dramatic_expressive: float = Field(..., ge=0, le=1)
    emotion_mood_optimistic_pleasant_dark: float = Field(..., ge=0, le=1)
    emotion_mood_low_energy_high_energy: float = Field(..., ge=0, le=1)
    emotion_mood_honest_imaginative: float = Field(..., ge=0, le=1)
    interpretation_unsatisfactory_convincing: float = Field(..., ge=0, le=1)


class TemporalAnalysisItem(BaseModel):
    """Analysis results for a single time chunk"""

    timestamp: str = Field(
        ..., description="Human-readable timestamp (e.g., '0:00-0:03')"
    )
    timestamp_seconds: List[float] = Field(
        ...,
        description="Start and end time in seconds [start, end]",
        min_items=2,
        max_items=2,
    )
    scores: DimensionScores = Field(
        ..., description="Perceptual dimension scores for this chunk"
    )
    alerts: List[AlertItem] = Field(
        default=[], description="Alerts for scores below thresholds"
    )


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis process"""

    total_duration: float = Field(..., description="Total audio duration in seconds")
    total_chunks: int = Field(..., description="Number of audio chunks analyzed")
    chunk_duration: float = Field(..., description="Duration of each chunk in seconds")
    overlap: float = Field(..., description="Overlap between chunks in seconds")


class AnalysisResponse(BaseModel):
    """Complete API response for piano performance analysis"""

    overall_scores: DimensionScores = Field(
        ..., description="Average scores across all chunks"
    )
    temporal_analysis: List[TemporalAnalysisItem] = Field(
        ..., description="Chunk-by-chunk analysis results"
    )
    metadata: AnalysisMetadata = Field(..., description="Analysis process metadata")


class AnalysisRequest(BaseModel):
    """Request parameters (used internally, not for multipart form)"""

    chunk_duration: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Duration of each analysis chunk in seconds",
    )
    overlap: float = Field(
        default=1.0, ge=0.0, description="Overlap between chunks in seconds"
    )
    thresholds: Optional[Dict[str, float]] = Field(
        default=None, description="Custom thresholds for alerts"
    )

    @root_validator(pre=True)
    def validate_overlap_and_chunk_duration(cls, values):
        chunk_duration = values.get("chunk_duration")
        overlap = values.get("overlap")

        if chunk_duration is None or overlap is None:
            raise ValueError("Both chunk_duration and overlap must be provided")

        if overlap >= chunk_duration:
            raise ValueError("overlap must be less than chunk_duration")

        return values


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(
        default=None, description="Detailed error information"
    )


class HealthResponse(BaseModel):
    """Health check response"""
    
    # Configure model to allow protected namespaces
    model_config = {'protected_namespaces': ()}
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")


# LLM-Enhanced Feedback Models
class LLMFeedbackInsight(BaseModel):
    """Individual insight in temporal feedback"""
    
    category: str = Field(..., description="Category: Technical|Musical|Interpretive")
    observation: str = Field(..., description="Specific observation about this time segment")
    actionable_advice: str = Field(..., description="Concrete practice suggestion or technique")
    score_reference: str = Field(..., description="Reference to dimension and score")


class LLMTemporalFeedback(BaseModel):
    """Temporal feedback for a specific time segment"""
    
    timestamp: str = Field(..., description="Time segment (e.g., '0:00-0:03')")
    insights: List[LLMFeedbackInsight] = Field(..., description="Specific insights for this segment")
    practice_focus: str = Field(..., description="Primary area to work on in this passage")


class LLMOverallAssessment(BaseModel):
    """Overall assessment of the performance"""
    
    strengths: List[str] = Field(..., description="2-3 key technical/musical strengths")
    priority_areas: List[str] = Field(..., description="2-3 most important areas to focus on")
    performance_character: str = Field(..., description="Brief description of overall interpretive character")


class LLMPracticeRecommendation(BaseModel):
    """Individual practice recommendation"""
    
    skill_area: str = Field(..., description="Name of technique/skill")
    specific_exercise: str = Field(..., description="Detailed practice method or exercise")
    expected_outcome: str = Field(..., description="What improvement to expect")


class LLMLongTermDevelopment(BaseModel):
    """Long-term development recommendation"""
    
    musical_aspect: str = Field(..., description="Broader musical concept")
    development_approach: str = Field(..., description="How to cultivate this aspect")
    repertoire_suggestions: str = Field(..., description="Types of pieces that would help")


class LLMPracticeRecommendations(BaseModel):
    """Complete practice recommendations"""
    
    immediate_priorities: List[LLMPracticeRecommendation] = Field(
        ..., description="Immediate technical priorities"
    )
    long_term_development: List[LLMLongTermDevelopment] = Field(
        ..., description="Long-term musical development"
    )


class LLMFeedback(BaseModel):
    """Complete LLM-generated pedagogical feedback"""
    
    # Configure model to allow field aliases and underscore fields
    model_config = {'populate_by_name': True}
    
    overall_assessment: LLMOverallAssessment = Field(..., description="Overall performance assessment")
    temporal_feedback: List[LLMTemporalFeedback] = Field(..., description="Time-specific feedback")
    practice_recommendations: LLMPracticeRecommendations = Field(..., description="Practice guidance")
    encouragement: str = Field(..., description="Motivating message acknowledging progress and potential")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Generation metadata", alias="_metadata")


class EnhancedAnalysisResponse(BaseModel):
    """Enhanced analysis response with LLM-generated pedagogical feedback"""
    
    # Original analysis data
    overall_scores: DimensionScores = Field(
        ..., description="Average scores across all chunks"
    )
    temporal_analysis: List[TemporalAnalysisItem] = Field(
        ..., description="Chunk-by-chunk analysis results"
    )
    metadata: AnalysisMetadata = Field(..., description="Analysis process metadata")
    
    # LLM-enhanced features
    llm_feedback: Optional[LLMFeedback] = Field(
        default=None, description="AI-generated pedagogical feedback (if available)"
    )
    llm_service_status: str = Field(
        default="unavailable", 
        description="Status of LLM service: available|unavailable|error"
    )
    filtered_scores_summary: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Summary of filtered scores (ignoring zeros, categorized as low/moderate/good)"
    )


# Example responses for documentation
example_analysis_response = {
    "overall_scores": {
        "timing_stable_unstable": 0.72,
        "articulation_short_long": 0.68,
        "articulation_soft_cushioned_hard_solid": 0.45,
        "pedal_sparse_dry_saturated_wet": 0.63,
        "pedal_clean_blurred": 0.71,
        "timbre_even_colorful": 0.58,
        "timbre_shallow_rich": 0.82,
        "timbre_bright_dark": 0.67,
        "timbre_soft_loud": 0.55,
        "dynamic_sophisticated_mellow_raw_crude": 0.73,
        "dynamic_little_range_large_range": 0.48,
        "music_making_fast_paced_slow_paced": 0.66,
        "music_making_flat_spacious": 0.59,
        "music_making_disproportioned_balanced": 0.77,
        "music_making_pure_dramatic_expressive": 0.84,
        "emotion_mood_optimistic_pleasant_dark": 0.62,
        "emotion_mood_low_energy_high_energy": 0.71,
        "emotion_mood_honest_imaginative": 0.69,
        "interpretation_unsatisfactory_convincing": 0.75,
    },
    "temporal_analysis": [
        {
            "timestamp": "0:00-0:03",
            "timestamp_seconds": [0, 3],
            "scores": {
                "timing_stable_unstable": 0.85,
                "articulation_short_long": 0.70,
                "articulation_soft_cushioned_hard_solid": 0.60,
                "pedal_sparse_dry_saturated_wet": 0.68,
                "pedal_clean_blurred": 0.75,
                "timbre_even_colorful": 0.62,
                "timbre_shallow_rich": 0.88,
                "timbre_bright_dark": 0.72,
                "timbre_soft_loud": 0.58,
                "dynamic_sophisticated_mellow_raw_crude": 0.78,
                "dynamic_little_range_large_range": 0.52,
                "music_making_fast_paced_slow_paced": 0.70,
                "music_making_flat_spacious": 0.63,
                "music_making_disproportioned_balanced": 0.82,
                "music_making_pure_dramatic_expressive": 0.89,
                "emotion_mood_optimistic_pleasant_dark": 0.67,
                "emotion_mood_low_energy_high_energy": 0.76,
                "emotion_mood_honest_imaginative": 0.73,
                "interpretation_unsatisfactory_convincing": 0.80,
            },
            "alerts": [],
        }
    ],
    "metadata": {
        "total_duration": 30.5,
        "total_chunks": 10,
        "chunk_duration": 3.0,
        "overlap": 1.0,
    },
}

example_error_response = {
    "error": "Unsupported file format",
    "detail": "Supported formats: .mp3, .wav, .flac, .m4a",
}
