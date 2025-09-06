"""
Model loading and inference for Piano Perception Transformer
"""

import os
import stat
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from flax import linen as nn
from typing import Optional, Dict, Any
import logging
from pathlib import Path

from .config import *

logger = logging.getLogger(__name__)


class ProductionASTForRegression(nn.Module):
    """AST model with regression head for perceptual prediction"""

    patch_size: int = PATCH_SIZE
    embed_dim: int = EMBED_DIM
    num_layers: int = NUM_LAYERS
    num_heads: int = NUM_HEADS
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    stochastic_depth_rate: float = 0.1
    num_outputs: int = NUM_OUTPUTS

    def setup(self):
        self.drop_rates = [
            self.stochastic_depth_rate * i / (self.num_layers - 1)
            for i in range(self.num_layers)
        ]

    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Forward pass for inference
        Args:
            x: Mel-spectrogram [batch, time, freq] -> (batch, 128, 128)
            training: Always False for inference
        Returns:
            predictions: [batch, 19] regression outputs
        """
        batch_size, time_frames, freq_bins = x.shape

        # === PATCH EMBEDDING ===
        patch_size = self.patch_size

        # Ensure input can be divided into patches
        time_pad = (patch_size - time_frames % patch_size) % patch_size
        freq_pad = (patch_size - freq_bins % patch_size) % patch_size

        if time_pad > 0 or freq_pad > 0:
            x = jnp.pad(
                x,
                ((0, 0), (0, time_pad), (0, freq_pad)),
                mode="constant",
                constant_values=-80.0,
            )

        time_patches = x.shape[1] // patch_size
        freq_patches = x.shape[2] // patch_size
        num_patches = time_patches * freq_patches

        # Reshape to patches: [batch, num_patches, patch_dim]
        x = x.reshape(batch_size, time_patches, patch_size, freq_patches, patch_size)
        x = x.transpose(0, 1, 3, 2, 4)
        x = x.reshape(batch_size, num_patches, patch_size * patch_size)

        # Linear patch embedding
        x = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            name="patch_embedding",
        )(x)

        # === POSITIONAL ENCODING ===
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, num_patches, self.embed_dim),
        )
        x = x + pos_embedding

        # No dropout during inference

        # === TRANSFORMER LAYERS ===
        for layer_idx in range(self.num_layers):
            # Multi-Head Self-Attention Block
            residual = x
            x = nn.LayerNorm(epsilon=1e-6, name=f"norm1_layer{layer_idx}")(x)

            attention = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=0.0,  # No dropout during inference
                kernel_init=nn.initializers.truncated_normal(stddev=0.02),
                bias_init=nn.initializers.zeros,
                name=f"attention_layer{layer_idx}",
            )(x, x, deterministic=True)

            x = residual + attention

            # Feed-Forward Network Block
            residual = x
            x = nn.LayerNorm(epsilon=1e-6, name=f"norm2_layer{layer_idx}")(x)

            # MLP with 4x expansion
            mlp_hidden = int(self.embed_dim * self.mlp_ratio)

            mlp = nn.Dense(
                mlp_hidden,
                kernel_init=nn.initializers.truncated_normal(stddev=0.02),
                bias_init=nn.initializers.zeros,
                name=f"mlp_dense1_layer{layer_idx}",
            )(x)
            mlp = nn.gelu(mlp)

            mlp = nn.Dense(
                self.embed_dim,
                kernel_init=nn.initializers.truncated_normal(stddev=0.02),
                bias_init=nn.initializers.zeros,
                name=f"mlp_dense2_layer{layer_idx}",
            )(mlp)

            x = residual + mlp

        # === FINAL PROCESSING ===
        x = nn.LayerNorm(epsilon=1e-6, name="final_norm")(x)

        # === REGRESSION HEAD ===
        # Global average pooling over patches
        x = jnp.mean(x, axis=1)  # [batch, embed_dim]

        # Regression layers
        x = nn.Dense(
            512,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            name="regression_hidden",
        )(x)
        x = nn.gelu(x)

        # Final prediction layer
        predictions = nn.Dense(
            self.num_outputs,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            name="regression_output",
        )(x)

        return predictions


class PianoPerceptionModel:
    """Wrapper class for the Piano Perception Transformer model"""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.params = None
        self.label_scaler = None
        self._loaded = False

    def load_model(self) -> bool:
        """Load the pre-trained model from checkpoint"""
        try:
            # TODO: Migrate to flax.serialization.msgpack or orbax checkpoint format
            # for safer model serialization instead of pickle

            try:
                # Get absolute real path, resolving any symlinks
                real_path = os.path.realpath(self.model_path)

                # Ensure path is under allowed models directory
                models_dir = os.path.realpath(os.path.dirname(MODEL_PATH))
                if not real_path.startswith(models_dir):
                    raise ValueError(f"Model path '{real_path}' is not within the allowed models directory '{models_dir}'")

                # Check file exists and is a regular file
                if not os.path.exists(real_path):
                    raise FileNotFoundError(f"Model file not found: {real_path}")

                if not os.path.isfile(real_path):
                    raise ValueError(f"Not a regular file: {real_path}")

                # Check permissions (should only be readable by owner/group)
                mode = os.stat(real_path).st_mode
                if mode & (stat.S_IWOTH | stat.S_IXOTH):
                    raise ValueError(f"Unsafe file permissions on {real_path}")

                logger.info(f"Loading validated model from: {real_path}")

                # Load checkpoint with validated path
                with open(real_path, "rb") as f:
                    checkpoint = pickle.load(f)

            except (ValueError, FileNotFoundError, OSError) as e:
                logger.error(f"Model validation failed: {e}")
                return False

            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                raise ValueError(
                    "Expected checkpoint to be a dictionary, " f"got {type(checkpoint)}"
                )

            if "params" not in checkpoint:
                raise ValueError("Checkpoint missing required 'params' key")

            # Extract components safely
            # Handle nested parameter structure if present
            params_raw = checkpoint["params"]
            if isinstance(params_raw, dict) and "params" in params_raw:
                # Parameters are nested one level deeper (checkpoint["params"]["params"])
                self.params = params_raw["params"]
            else:
                # Parameters are at the expected level (checkpoint["params"])
                self.params = params_raw
            self.label_scaler = checkpoint.get("label_scaler", None)

            # Initialize model architecture
            self.model = ProductionASTForRegression()

            # Validate parameter structure and shape
            if not isinstance(self.params, dict):
                raise ValueError(
                    "Model parameters must be a dictionary, " f"got {type(self.params)}"
                )

            # Get expected parameter structure
            dummy_input = jnp.ones((1, 128, 128))
            try:
                key = jax.random.PRNGKey(0)
                variables = self.model.init(key, dummy_input, training=False)
                expected_params = variables["params"]
                expected_struct = jtu.tree_structure(expected_params)
                actual_struct = jtu.tree_structure(self.params)
                if expected_struct != actual_struct:
                    raise ValueError(
                        "Parameter structure mismatch. This may indicate an "
                        "incompatible or corrupted checkpoint. Expected structure "
                        "does not match loaded parameters."
                    )
            except Exception as e:
                raise ValueError(f"Failed to validate parameter structure: {str(e)}")

            # Verify parameter count
            param_count = sum(x.size for x in jtu.tree_leaves(self.params))
            logger.info(f"Model loaded successfully: {param_count:,} parameters")

            # Test model with dummy input
            dummy_input = jnp.ones((1, 128, 128))
            try:
                # Test forward pass to validate params
                output = self.model.apply(
                    {"params": self.params}, dummy_input, training=False
                )
                expected_shape = (1, self.model.num_outputs)
                actual_shape = output.shape
                if actual_shape != expected_shape:
                    raise ValueError(
                        "Unexpected output shape: "
                        f"expected {expected_shape}, got {actual_shape}"
                    )
                logger.info("Model validation successful")
            except Exception as e:
                logger.error(f"Model validation failed: {str(e)}")
                return False

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, spectrograms: np.ndarray) -> np.ndarray:
        """
        Predict perceptual dimensions for batch of spectrograms

        Args:
            spectrograms: Array of shape [batch, 128, 128]

        Returns:
            predictions: Array of shape [batch, 19] (normalized)
        """
        if not self._loaded:
            if not self.load_model():
                raise RuntimeError("Model not loaded")

        # Convert to JAX array
        if not isinstance(spectrograms, jnp.ndarray):
            spectrograms = jnp.array(spectrograms)

        # Ensure correct shape
        if len(spectrograms.shape) == 2:
            spectrograms = spectrograms[None, ...]  # Add batch dimension

        if spectrograms.shape[1:] != TARGET_SHAPE:
            raise ValueError(
                f"Expected shape [batch, 128, 128], got {spectrograms.shape}"
            )

        # Model inference
        try:
            # Forward pass with properly structured parameters
            predictions = self.model.apply(
                {"params": self.params}, spectrograms, training=False
            )
            return np.array(predictions)
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")

    def predict_single(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Predict for a single spectrogram

        Args:
            spectrogram: Array of shape [128, 128]

        Returns:
            prediction: Array of shape [19] (normalized)
        """
        batch_pred = self.predict(spectrogram[None, ...])
        return batch_pred[0]

    def denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Denormalize predictions back to original scale

        Args:
            predictions: Normalized predictions [batch, 19] or [19]

        Returns:
            denormalized: Original scale predictions
        """
        if self.label_scaler is None:
            logger.warning(
                "No label scaler available - returning normalized predictions"
            )
            return predictions

        try:
            return self.label_scaler.inverse_transform(predictions)
        except Exception as e:
            logger.warning(f"Denormalization failed: {e}")
            return predictions

    def get_dimension_names(self) -> list:
        """Get list of dimension names for API response"""
        return API_DIMENSION_NAMES.copy()

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._loaded


# Global model instance (singleton pattern for API)
_global_model: Optional[PianoPerceptionModel] = None


def get_model() -> PianoPerceptionModel:
    """Get global model instance (lazy loading)"""
    global _global_model
    if _global_model is None:
        _global_model = PianoPerceptionModel()
        if not _global_model.load_model():
            raise RuntimeError("Failed to load Piano Perception model")
    return _global_model


def load_model_on_startup(model_path: str = MODEL_PATH) -> bool:
    """Load model during API startup"""
    try:
        global _global_model
        _global_model = PianoPerceptionModel(model_path)
        return _global_model.load_model()
    except Exception as e:
        logger.error(f"Startup model loading failed: {e}")
        return False
