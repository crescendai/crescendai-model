"""
Configuration settings for Piano Performance Analysis API
"""

import os
from typing import Dict, List

# API Configuration
API_TITLE = "Piano Performance Analysis API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Analyze piano performance across 19 perceptual dimensions using AST"

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "./models/final_finetuned_model.pkl")
EMBED_DIM = 768
NUM_LAYERS = 12
NUM_HEADS = 12
PATCH_SIZE = 16
NUM_OUTPUTS = 19

# Audio Processing Configuration
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
TARGET_SHAPE = (128, 128)  # (time, freq)

# Analysis Configuration
MAX_DURATION = 30.0  # seconds
DEFAULT_CHUNK_DURATION = 3.0  # seconds
DEFAULT_OVERLAP = 1.0  # seconds
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Default thresholds for alerts
DEFAULT_THRESHOLDS = {
    "timing_stable_unstable": 0.6,
    "articulation_short_long": 0.5,
    "articulation_soft_cushioned_hard_solid": 0.5,
    "pedal_sparse_dry_saturated_wet": 0.5,
    "pedal_clean_blurred": 0.5,
    "timbre_even_colorful": 0.5,
    "timbre_shallow_rich": 0.5,
    "timbre_bright_dark": 0.5,
    "timbre_soft_loud": 0.5,
    "dynamic_sophisticated_mellow_raw_crude": 0.55,
    "dynamic_little_range_large_range": 0.5,
    "music_making_fast_paced_slow_paced": 0.5,
    "music_making_flat_spacious": 0.5,
    "music_making_disproportioned_balanced": 0.5,
    "music_making_pure_dramatic_expressive": 0.5,
    "emotion_mood_optimistic_pleasant_dark": 0.5,
    "emotion_mood_low_energy_high_energy": 0.5,
    "emotion_mood_honest_imaginative": 0.5,
    "interpretation_unsatisfactory_convincing": 0.5
}

# 19 Perceptual Dimensions (matching training order)
DIMENSION_NAMES = [
    "Timing_Stable_Unstable",
    "Articulation_Short_Long", 
    "Articulation_Soft_cushioned_Hard_solid",
    "Pedal_Sparse/dry_Saturated/wet",
    "Pedal_Clean_Blurred",
    "Timbre_Even_Colorful",
    "Timbre_Shallow_Rich",
    "Timbre_Bright_Dark",
    "Timbre_Soft_Loud",
    "Dynamic_Sophisticated/mellow_Raw/crude",
    "Dynamic_Little_dynamic_range_Large_dynamic_range",
    "Music_Making_Fast_paced_Slow_paced",
    "Music_Making_Flat_Spacious",
    "Music_Making_Disproportioned_Balanced",
    "Music_Making_Pure_Dramatic/expressive",
    "Emotion_&_Mood_Optimistic/pleasant_Dark",
    "Emotion_&_Mood_Low_Energy_High_Energy",
    "Emotion_&_Mood_Honest_Imaginative",
    "Interpretation_Unsatisfactory/doubtful_Convincing"
]

# API response field names (cleaned up for JSON)
API_DIMENSION_NAMES = [
    "timing_stable_unstable",
    "articulation_short_long",
    "articulation_soft_cushioned_hard_solid",
    "pedal_sparse_dry_saturated_wet",
    "pedal_clean_blurred",
    "timbre_even_colorful",
    "timbre_shallow_rich",
    "timbre_bright_dark",
    "timbre_soft_loud",
    "dynamic_sophisticated_mellow_raw_crude",
    "dynamic_little_range_large_range",
    "music_making_fast_paced_slow_paced",
    "music_making_flat_spacious",
    "music_making_disproportioned_balanced",
    "music_making_pure_dramatic_expressive",
    "emotion_mood_optimistic_pleasant_dark",
    "emotion_mood_low_energy_high_energy",
    "emotion_mood_honest_imaginative",
    "interpretation_unsatisfactory_convincing"
]

# Supported file formats
SUPPORTED_FORMATS = [".mp3", ".wav", ".flac", ".m4a"]

# Error messages
ERROR_MESSAGES = {
    "invalid_format": f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}",
    "file_too_large": f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024*1024)}MB",
    "duration_too_long": f"Audio duration exceeds maximum limit of {MAX_DURATION} seconds",
    "model_not_found": f"Model file not found at {MODEL_PATH}",
    "invalid_chunk_duration": "Chunk duration must be between 1.0 and 10.0 seconds",
    "invalid_overlap": "Overlap must be between 0.0 and chunk_duration seconds"
}