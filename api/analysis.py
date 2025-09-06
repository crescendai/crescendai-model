"""
Analysis logic and alert system for Piano Performance Analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime

from core.config import DEFAULT_THRESHOLDS, API_DIMENSION_NAMES
from .models import (
    DimensionScores, TemporalAnalysisItem, AnalysisMetadata, 
    AnalysisResponse, AlertItem, AlertSeverity
)

logger = logging.getLogger(__name__)

def predictions_to_dimension_scores(predictions: np.ndarray) -> DimensionScores:
    """
    Convert model predictions to DimensionScores object
    
    Args:
        predictions: Array of shape [19] with prediction scores
        
    Returns:
        DimensionScores object with named fields
    """
    if len(predictions) != 19:
        raise ValueError(f"Expected 19 predictions, got {len(predictions)}")
    
    # Ensure scores are in valid range [0, 1]
    predictions = np.clip(predictions, 0.0, 1.0)
    
    # Map to dimension names
    scores_dict = {}
    for i, name in enumerate(API_DIMENSION_NAMES):
        scores_dict[name] = float(predictions[i])
    
    return DimensionScores(**scores_dict)

def calculate_overall_scores(all_predictions: List[np.ndarray]) -> DimensionScores:
    """
    Calculate overall scores by averaging across all chunks
    
    Args:
        all_predictions: List of prediction arrays, each shape [19]
        
    Returns:
        DimensionScores with averaged predictions
    """
    if not all_predictions:
        raise ValueError("No predictions provided")
    
    # Stack and average predictions
    stacked = np.stack(all_predictions, axis=0)  # Shape: [num_chunks, 19]
    averaged = np.mean(stacked, axis=0)  # Shape: [19]
    
    return predictions_to_dimension_scores(averaged)

def generate_alerts(scores: DimensionScores, thresholds: Dict[str, float]) -> List[AlertItem]:
    """
    Generate alerts for scores below thresholds
    
    Args:
        scores: Dimension scores to check
        thresholds: Threshold values for each dimension
        
    Returns:
        List of alert items
    """
    alerts = []
    
    for dimension, threshold in thresholds.items():
        if hasattr(scores, dimension):
            score = getattr(scores, dimension)
            
            if score < threshold:
                # Determine severity based on how far below threshold
                diff = threshold - score
                if diff >= 0.3:
                    severity = AlertSeverity.HIGH
                elif diff >= 0.15:
                    severity = AlertSeverity.MODERATE
                else:
                    severity = AlertSeverity.LOW
                
                alerts.append(AlertItem(
                    dimension=dimension,
                    score=score,
                    threshold=threshold,
                    severity=severity
                ))
    
    return alerts

def format_timestamp(start_time: float, end_time: float) -> str:
    """
    Format timestamp as human-readable string
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Formatted timestamp string (e.g., "1:23-1:26")
    """
    def seconds_to_mmss(seconds):
        minutes = int(seconds // 60)
        secs = round(seconds % 60)
        # Handle edge case where rounding to 60 seconds
        if secs == 60:
            minutes += 1
            secs = 0
        return f"{minutes}:{secs:02d}"
    
    start_str = seconds_to_mmss(start_time)
    end_str = seconds_to_mmss(end_time)
    return f"{start_str}-{end_str}"

def analyze_piano_performance(
    all_predictions: List[np.ndarray],
    timestamps: List[Tuple[float, float]],
    chunk_duration: float,
    overlap: float,
    total_duration: float,
    thresholds: Optional[Dict[str, float]] = None
) -> AnalysisResponse:
    """
    Perform complete analysis of piano performance predictions
    
    Args:
        all_predictions: List of prediction arrays from model
        timestamps: List of (start, end) timestamp pairs
        chunk_duration: Duration of each chunk
        overlap: Overlap between chunks
        total_duration: Total audio duration
        thresholds: Custom thresholds for alerts (None = use defaults)
        
    Returns:
        Complete analysis response
    """
    if not all_predictions:
        raise ValueError("No predictions provided")
    
    if len(all_predictions) != len(timestamps):
        raise ValueError(f"Predictions ({len(all_predictions)}) and timestamps ({len(timestamps)}) count mismatch")
    
    # Use default thresholds if not provided
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()
    
    logger.info(f"Analyzing {len(all_predictions)} chunks with {len(thresholds)} threshold rules")
    
    try:
        # Calculate overall scores
        overall_scores = calculate_overall_scores(all_predictions)
        
        # Process each chunk for temporal analysis
        temporal_analysis = []
        
        for i, (predictions, (start_time, end_time)) in enumerate(zip(all_predictions, timestamps)):
            # Convert predictions to scores
            chunk_scores = predictions_to_dimension_scores(predictions)
            
            # Generate alerts for this chunk
            chunk_alerts = generate_alerts(chunk_scores, thresholds)
            
            # Create temporal analysis item
            temporal_item = TemporalAnalysisItem(
                timestamp=format_timestamp(start_time, end_time),
                timestamp_seconds=[start_time, end_time],
                scores=chunk_scores,
                alerts=chunk_alerts
            )
            
            temporal_analysis.append(temporal_item)
        
        # Create metadata
        metadata = AnalysisMetadata(
            total_duration=total_duration,
            total_chunks=len(all_predictions),
            chunk_duration=chunk_duration,
            overlap=overlap
        )
        
        # Create final response
        response = AnalysisResponse(
            overall_scores=overall_scores,
            temporal_analysis=temporal_analysis,
            metadata=metadata
        )
        
        # Log analysis summary
        total_alerts = sum(len(item.alerts) for item in temporal_analysis)
        high_alerts = sum(1 for item in temporal_analysis 
                         for alert in item.alerts if alert.severity == AlertSeverity.HIGH)
        
        logger.info(f"Analysis complete: {total_alerts} total alerts, {high_alerts} high severity")
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise ValueError(f"Analysis failed: {e}")

def get_performance_summary(response: AnalysisResponse) -> Dict[str, Any]:
    """
    Generate a performance summary for logging/debugging
    
    Args:
        response: Analysis response
        
    Returns:
        Dictionary with summary statistics
    """
    try:
        # Count alerts by severity
        alert_counts = {"high": 0, "moderate": 0, "low": 0, "total": 0}
        
        for item in response.temporal_analysis:
            for alert in item.alerts:
                alert_counts[alert.severity.value] += 1
                alert_counts["total"] += 1
        
        # Get overall score statistics
        overall_dict = response.overall_scores.dict()
        scores_array = np.array(list(overall_dict.values()))
        
        summary = {
            "total_chunks": response.metadata.total_chunks,
            "total_duration": response.metadata.total_duration,
            "alert_counts": alert_counts,
            "overall_score_stats": {
                "mean": float(np.mean(scores_array)),
                "std": float(np.std(scores_array)),
                "min": float(np.min(scores_array)),
                "max": float(np.max(scores_array))
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return summary
        
    except Exception as e:
        logger.warning(f"Failed to generate performance summary: {e}")
        return {"error": str(e)}

def validate_thresholds(thresholds: Dict[str, float]) -> Dict[str, float]:
    """
    Validate and clean custom thresholds
    
    Args:
        thresholds: User-provided thresholds
        
    Returns:
        Validated thresholds dictionary
        
    Raises:
        ValueError: If thresholds are invalid
    """
    if not isinstance(thresholds, dict):
        raise ValueError("Thresholds must be a dictionary")
    
    validated = {}
    valid_dimensions = set(API_DIMENSION_NAMES)
    
    for dimension, threshold in thresholds.items():
        # Check dimension name
        if dimension not in valid_dimensions:
            logger.warning(f"Unknown dimension in thresholds: {dimension}")
            continue
        
        # Check threshold value
        try:
            threshold_float = float(threshold)
            if not (0.0 <= threshold_float <= 1.0):
                raise ValueError(f"Threshold for {dimension} must be between 0 and 1, got {threshold_float}")
            validated[dimension] = threshold_float
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid threshold value for {dimension}: {threshold} - {e}")
    
    # Fill in missing dimensions with defaults
    for dimension in API_DIMENSION_NAMES:
        if dimension not in validated:
            validated[dimension] = DEFAULT_THRESHOLDS.get(dimension, 0.5)
    
    return validated