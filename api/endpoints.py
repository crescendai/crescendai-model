"""
FastAPI route handlers for Piano Performance Analysis API
"""

import os
import json
from typing import Optional, Dict
from datetime import datetime
import logging

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from core.model_loader import get_model
from core.audio_processing import (
    save_uploaded_file, cleanup_temp_file, validate_audio_file,
    process_audio_file, calculate_timestamps
)
from core.config import API_VERSION, MAX_FILE_SIZE, SUPPORTED_FORMATS, ERROR_MESSAGES
from .analysis import analyze_piano_performance as analyze_performance, validate_thresholds, get_performance_summary
from .models import AnalysisResponse, ErrorResponse, HealthResponse, example_analysis_response

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

def parse_thresholds(thresholds_json: Optional[str] = None) -> Optional[Dict[str, float]]:
    """Parse thresholds from JSON string"""
    if not thresholds_json:
        return None
    
    try:
        thresholds = json.loads(thresholds_json)
        return validate_thresholds(thresholds)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in thresholds: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post(
    "/analyze-piano-performance",
    response_model=AnalysisResponse,
    summary="Analyze Piano Performance",
    description="""
    Analyze a piano performance recording across 19 perceptual dimensions.
    
    **Input Requirements:**
    - Audio file: MP3, WAV, FLAC, or M4A format
    - Maximum file size: 10MB
    - Maximum duration: 30 seconds
    
    **Parameters:**
    - `chunk_duration`: Duration of each analysis chunk (1.0-10.0 seconds, default: 3.0)
    - `overlap`: Overlap between chunks (0.0 to chunk_duration, default: 1.0) 
    - `thresholds`: Optional JSON object with custom alert thresholds for dimensions
    
    **Returns:**
    - Overall performance scores averaged across all chunks
    - Temporal analysis with scores for each time chunk
    - Alerts for dimensions below specified thresholds
    - Analysis metadata (duration, chunks, etc.)
    """,
    responses={
        200: {
            "description": "Successful analysis",
            "content": {
                "application/json": {
                    "example": example_analysis_response
                }
            }
        },
        400: {"model": ErrorResponse, "description": "Invalid input"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def analyze_piano_performance(
    audio: UploadFile = File(..., description="Audio file to analyze (MP3/WAV/FLAC/M4A)"),
    chunk_duration: float = Form(default=3.0, ge=1.0, le=10.0, description="Chunk duration in seconds"),
    overlap: float = Form(default=1.0, ge=0.0, description="Overlap between chunks in seconds"),
    thresholds: Optional[str] = Form(default=None, description="JSON object with custom thresholds")
):
    """Main endpoint for piano performance analysis"""
    temp_file_path = None
    
    try:
        # Validate overlap relative to chunk_duration
        if overlap >= chunk_duration:
            raise HTTPException(
                status_code=400, 
                detail="Overlap must be less than chunk_duration"
            )
        
        # Read and validate file
        file_content = await audio.read()
        validate_audio_file(file_content, audio.filename)
        
        logger.info(f"Processing file: {audio.filename} ({len(file_content)} bytes)")
        
        # Parse thresholds if provided
        threshold_dict = parse_thresholds(thresholds)
        
        # Save uploaded file to temporary location
        temp_file_path = save_uploaded_file(file_content, audio.filename)
        
        # Process audio file
        spectrograms, total_duration = process_audio_file(
            temp_file_path, 
            chunk_duration=chunk_duration,
            overlap=overlap
        )
        
        # Calculate timestamps for chunks
        timestamps = calculate_timestamps(total_duration, chunk_duration, overlap)
        
        # Ensure we have matching spectrograms and timestamps
        # Ensure we have matching spectrograms and timestamps
        if len(spectrograms) != len(timestamps):
            logger.warning(
                f"Length mismatch: {len(spectrograms)} spectrograms vs {len(timestamps)} timestamps. "
                f"Truncating to {min(len(spectrograms), len(timestamps))} items."
            )
            # Trim to match (can happen due to audio processing edge cases)
            min_len = min(len(spectrograms), len(timestamps))
            spectrograms = spectrograms[:min_len]
            timestamps = timestamps[:min_len]
        # Get the model
        model = get_model()
        
        # Batch predict on all spectrograms with controlled batch size
        import numpy as np
        MAX_BATCH_SIZE = 32  # Adjust based on model and memory constraints

        all_predictions = []
        for i in range(0, len(spectrograms), MAX_BATCH_SIZE):
            batch = spectrograms[i : i + MAX_BATCH_SIZE]
            spectrogram_batch = np.array(batch)
            batch_predictions = model.predict(spectrogram_batch)
            all_predictions.extend(batch_predictions)
        
        # Split predictions back into individual chunks
        prediction_list = [all_predictions[i] for i in range(len(all_predictions))]
        
        # Perform analysis
        analysis_result = analyze_performance(
            all_predictions=prediction_list,
            timestamps=timestamps,
            chunk_duration=chunk_duration,
            overlap=overlap,
            total_duration=total_duration,
            thresholds=threshold_dict
        )
        
        # Log performance summary
        summary = get_performance_summary(analysis_result)
        logger.info(f"Analysis summary: {summary}")
        
        return analysis_result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"Analysis failed for {audio.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
        
    finally:
        # Cleanup temporary file
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API service and model are ready"
)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        model = get_model()
        model_loaded = model.is_loaded()
        
        return HealthResponse(
            status="healthy" if model_loaded else "model_not_loaded",
            model_loaded=model_loaded,
            timestamp=datetime.now().isoformat(),
            version=API_VERSION
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            timestamp=datetime.now().isoformat(),
            version=API_VERSION
        )

@router.get(
    "/info",
    summary="API Information",
    description="Get information about the API capabilities and configuration"
)
async def get_api_info():
    """Get API information"""
    return {
        "name": "Piano Performance Analysis API",
        "version": API_VERSION,
        "description": "Analyze piano performance across 19 perceptual dimensions using Audio Spectrogram Transformer",
        "model_info": {
            "architecture": "12-layer Audio Spectrogram Transformer with regression head",
            "parameters": "~86M",
            "input_format": "128x128 mel-spectrograms",
            "output_dimensions": 19
        },
        "supported_formats": SUPPORTED_FORMATS,
        "limits": {
            "max_file_size": f"{MAX_FILE_SIZE // (1024*1024)}MB",
            "max_duration": "30 seconds",
            "chunk_duration_range": "1.0-10.0 seconds"
        },
        "perceptual_dimensions": [
            "Timing Stability", "Articulation Length", "Articulation Touch", 
            "Pedal Usage", "Pedal Clarity", "Timbre Color", "Timbre Richness",
            "Timbre Brightness", "Dynamic Range", "Dynamic Sophistication",
            "Dynamic Variation", "Musical Pacing", "Spatial Quality",
            "Balance", "Expression", "Emotional Valence", "Energy Level",
            "Interpretive Authenticity", "Overall Convincingness"
        ],
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
