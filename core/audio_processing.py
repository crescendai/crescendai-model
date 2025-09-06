"""
Audio processing utilities for Piano Performance Analysis API
"""

import librosa
import numpy as np
import tempfile
import os
from typing import List, Tuple, Optional
import logging

from .config import *

logger = logging.getLogger(__name__)


def load_audio_file(
    file_path: str, max_duration: float = MAX_DURATION
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to numpy array

    Args:
        file_path: Path to audio file
        max_duration: Maximum duration in seconds

    Returns:
        audio: Audio samples as numpy array
        sample_rate: Sample rate
    """
    try:
        # Load audio with librosa (automatically handles different formats)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=max_duration)

        if len(y) == 0:
            raise ValueError("Empty audio file")

        # Ensure minimum duration (pad with silence if needed)
        min_samples = int(2.0 * SAMPLE_RATE)  # 2 seconds minimum
        if len(y) < min_samples:
            padding = min_samples - len(y)
            y = np.pad(y, (0, padding), mode="constant", constant_values=0.0)

        logger.info(f"Audio loaded: {len(y)} samples, {len(y)/sr:.2f}s")
        return y, sr

    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        raise ValueError(f"Could not load audio file: {e}")


def audio_to_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Convert audio to mel-spectrogram with fixed dimensions

    Args:
        audio: Audio samples
        sr: Sample rate

    Returns:
        spectrogram: Mel-spectrogram [time, freq] with shape (128, 128)
    """
    try:
        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            power=2.0,
            fmin=20,
            fmax=sr // 2,
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to [time, freq] format
        mel_spec_transposed = mel_spec_db.T

        # Normalize to target shape (128, 128)
        target_time, target_freq = TARGET_SHAPE
        current_time, current_freq = mel_spec_transposed.shape

        # Handle time dimension
        if current_time >= target_time:
            # Truncate to target length
            normalized_spec = mel_spec_transposed[:target_time, :]
        else:
            # Pad to target length with silence
            pad_width = target_time - current_time
            normalized_spec = np.pad(
                mel_spec_transposed,
                ((0, pad_width), (0, 0)),
                mode="constant",
                constant_values=-80.0,
            )

        # Handle frequency dimension
        if normalized_spec.shape[1] >= target_freq:
            # Truncate
            normalized_spec = normalized_spec[:, :target_freq]
        else:
            # Pad
            pad_width = target_freq - normalized_spec.shape[1]
            normalized_spec = np.pad(
                normalized_spec,
                ((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=-80.0,
            )

        # Verify final shape
        if normalized_spec.shape != TARGET_SHAPE:
            raise ValueError(
                f"Spectrogram shape mismatch: {normalized_spec.shape} != {TARGET_SHAPE}"
            )

        return normalized_spec.astype(np.float32)

    except Exception as e:
        logger.error(f"Spectrogram conversion failed: {e}")
        raise ValueError(f"Could not convert to spectrogram: {e}")


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    overlap: float = DEFAULT_OVERLAP,
) -> List[np.ndarray]:
    """
    Split audio into overlapping chunks

    Args:
        audio: Audio samples
        sr: Sample rate
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds

    Returns:
        chunks: List of audio chunks
    """
    if chunk_duration <= 0 or chunk_duration > 10.0:
        raise ValueError("Chunk duration must be between 0 and 10 seconds")

    if overlap < 0 or overlap >= chunk_duration:
        raise ValueError("Overlap must be between 0 and chunk_duration")

    chunk_samples = int(chunk_duration * sr)
    hop_samples = int((chunk_duration - overlap) * sr)

    if hop_samples <= 0:
        hop_samples = chunk_samples  # No overlap

    chunks = []
    start = 0

    while start + chunk_samples <= len(audio):
        chunk = audio[start : start + chunk_samples]
        chunks.append(chunk)
        start += hop_samples

    # Handle remaining audio (if any)
    if start < len(audio) and len(chunks) > 0:
        # Pad the last chunk to full length
        remaining = audio[start:]
        padding_needed = chunk_samples - len(remaining)
        if padding_needed > 0:
            remaining = np.pad(
                remaining, (0, padding_needed), mode="constant", constant_values=0.0
            )
        chunks.append(remaining)
    elif len(chunks) == 0:
        # Audio is shorter than chunk duration - pad it
        padding_needed = chunk_samples - len(audio)
        padded_audio = np.pad(
            audio, (0, padding_needed), mode="constant", constant_values=0.0
        )
        chunks.append(padded_audio)

    logger.info(f"Audio chunked into {len(chunks)} chunks of {chunk_duration}s each")
    return chunks


def process_audio_file(
    file_path: str,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    overlap: float = DEFAULT_OVERLAP,
    max_duration: float = MAX_DURATION,
) -> Tuple[List[np.ndarray], float]:
    """
    Complete audio processing pipeline: load -> chunk -> spectrogram

    Args:
        file_path: Path to audio file
        chunk_duration: Duration of each chunk
        overlap: Overlap between chunks
        max_duration: Maximum total duration

    Returns:
        spectrograms: List of spectrogram arrays [128, 128]
        total_duration: Total audio duration processed
    """
    try:
        # Load audio
        audio, sr = load_audio_file(file_path, max_duration)
        total_duration = len(audio) / sr

        # Chunk audio
        chunks = chunk_audio(audio, sr, chunk_duration, overlap)

        # Convert each chunk to spectrogram
        spectrograms = []
        for i, chunk in enumerate(chunks):
            try:
                spectrogram = audio_to_spectrogram(chunk, sr)
                spectrograms.append(spectrogram)
            except Exception as e:
                logger.warning(f"Failed to process chunk {i}: {e}")
                # Skip this chunk but continue with others
                continue

        if len(spectrograms) == 0:
            raise ValueError("No spectrograms could be generated from audio")

        logger.info(
            f"Generated {len(spectrograms)} spectrograms from {total_duration:.2f}s audio"
        )
        return spectrograms, total_duration

    except Exception as e:
        logger.error(f"Audio processing pipeline failed: {e}")
        raise


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """
    Save uploaded file to temporary location

    Args:
        file_content: File content as bytes
        filename: Original filename

    Returns:
        temp_path: Path to saved temporary file
    """
    try:
        # Get file extension
        _, ext = os.path.splitext(filename.lower())

        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        logger.info(f"Saved uploaded file to: {temp_path}")
        return temp_path

    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise


def cleanup_temp_file(file_path: str) -> None:
    """Remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


def validate_audio_file(file_content: bytes, filename: str) -> None:
    """
    Validate uploaded audio file

    Args:
        file_content: File content
        filename: Original filename

    Raises:
        ValueError: If validation fails
    """
    # Check file size
    if len(file_content) > MAX_FILE_SIZE:
        raise ValueError(ERROR_MESSAGES["file_too_large"])

    # Check file extension
    _, ext = os.path.splitext(filename.lower())
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(ERROR_MESSAGES["invalid_format"])

    # Check if file content is not empty
    if len(file_content) == 0:
        raise ValueError("Empty file uploaded")


def calculate_timestamps(
    total_duration: float,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    overlap: float = DEFAULT_OVERLAP,
) -> List[Tuple[float, float]]:
    """
    Calculate timestamp ranges for each chunk

    Args:
        total_duration: Total audio duration in seconds
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds

    Returns:
        timestamps: List of (start, end) timestamp pairs

    Raises:
        ValueError: If chunk duration or overlap parameters are invalid
    """
    # Convert inputs to float for consistency
    total_duration = float(total_duration)
    chunk_duration = float(chunk_duration)
    overlap = float(overlap)

    # Validate inputs
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_duration:
        raise ValueError("overlap must be less than chunk_duration")

    timestamps = []
    hop_duration = chunk_duration - overlap
    current_start = 0.0

    while current_start < total_duration:
        current_end = min(current_start + chunk_duration, total_duration)
        timestamps.append((current_start, current_end))
        current_start += hop_duration

        # Avoid infinite loop for edge cases
        if current_start >= total_duration:
            break

    return timestamps
