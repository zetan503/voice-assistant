"""
Text-to-speech module for voice assistant.
Provides functionality for converting text to speech with streaming support.
"""
import logging
import os
import tempfile
import asyncio
import numpy as np
from typing import Dict, Generator, Optional, Union

import torch
from TTS.api import TTS

from config.settings import MODELS, AUDIO

logger = logging.getLogger(__name__)

class TextToSpeech:
    """
    Text-to-speech synthesis using TTS library.
    Supports real-time speech synthesis with streaming output.
    """

    def __init__(self, model_config: Optional[Dict] = None):
        """
        Initialize the TTS model with optional custom configuration.

        Args:
            model_config: Optional custom model configuration
        """
        self.config = model_config or MODELS["tts"]
        self.device = self.config["device"]
        self.sample_rate = AUDIO["sample_rate"]
        self.model = None

        # Validate device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        self.load_model()

    def load_model(self) -> None:
        """Load the TTS model based on configuration."""
        try:
            logger.info(f"Loading TTS model: {self.config['model_id']}")

            # Initialize TTS with the specified model
            self.model = TTS(
                model_name=self.config['model_id'],
                gpu=(self.device == "cuda")
            )

            logger.info(f"TTS model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load TTS model: {str(e)}")
            raise

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Text to be converted to speech

        Returns:
            NumPy array of audio samples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        try:
            # Generate speech
            wav = self.model.tts(
                text=text,
                speaker_name=None,  # Use default speaker
                language="en"
            )

            # Return as numpy array
            return np.array(wav)

        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            return np.array([], dtype=np.float32)

    def synthesize_to_file(self, text: str, output_path: str) -> str:
        """
        Synthesize speech from text and save to a file.

        Args:
            text: Text to be converted to speech
            output_path: Path to save the audio file

        Returns:
            Path to the saved audio file
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        try:
            # Generate speech and save to file
            self.model.tts_to_file(
                text=text,
                speaker_name=None,  # Use default speaker
                language="en",
                file_path=output_path
            )

            return output_path

        except Exception as e:
            logger.error(f"Error synthesizing speech to file: {str(e)}")
            return ""

    async def synthesize_stream(
        self,
        text: str,
        chunk_size: int = 1024
    ) -> Generator[np.ndarray, None, None]:
        """
        Synthesize speech from text and stream it in chunks.

        Args:
            text: Text to be converted to speech
            chunk_size: Size of audio chunks to yield

        Yields:
            Audio chunks as NumPy arrays
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        try:
            # Generate speech
            audio = self.synthesize(text)

            # Yield audio in chunks
            for i in range(0, len(audio), chunk_size):
                yield audio[i:i + chunk_size]
                # Small delay to simulate real-time streaming
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in stream synthesis: {str(e)}")
            yield np.array([], dtype=np.float32)