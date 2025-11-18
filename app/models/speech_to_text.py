"""
Speech-to-text module using faster-whisper for voice assistant.
Provides functionality for transcribing audio to text with streaming support.
"""
import logging
import numpy as np
import torch
from typing import Dict, Generator, List, Optional, Tuple, Union

from faster_whisper import WhisperModel

from config.settings import MODELS, AUDIO

logger = logging.getLogger(__name__)

class SpeechToText:
    """
    Speech-to-text transcription using Whisper model.
    Supports real-time transcription with streaming audio input.
    """

    def __init__(self, model_config: Optional[Dict] = None):
        """
        Initialize the STT model with optional custom configuration.

        Args:
            model_config: Optional custom model configuration
        """
        self.config = model_config or MODELS["stt"]
        self.device = self.config["device"]
        self.compute_type = self.config["compute_type"]
        self.sample_rate = AUDIO["sample_rate"]
        self.model = None

        # Validate device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.compute_type = "float32"

        self.load_model()

    def load_model(self) -> None:
        """Load the Whisper model based on configuration."""
        try:
            logger.info(f"Loading Whisper model: {self.config['model_id']}")

            # Load faster-whisper model
            self.model = WhisperModel(
                model_size_or_path=self.config['model_id'],
                device=self.device,
                compute_type=self.compute_type
            )

            logger.info(f"Whisper model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

    def transcribe(self, audio_data: Union[np.ndarray, str]) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: NumPy array of audio samples or path to audio file

        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        try:
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Combine all segment text
            result = " ".join(segment.text for segment in segments)

            return result.strip()

        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return ""

    async def transcribe_stream(
        self,
        audio_stream: Generator[np.ndarray, None, None]
    ) -> Generator[str, None, None]:
        """
        Transcribe streaming audio data asynchronously.

        Args:
            audio_stream: Generator yielding audio chunks

        Yields:
            Partial transcriptions as they become available
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        try:
            # Buffer for accumulating audio
            audio_buffer = np.array([], dtype=np.float32)

            # Process each chunk from the stream
            async for chunk in audio_stream:
                # Add chunk to buffer
                audio_buffer = np.append(audio_buffer, chunk)

                # Only process if we have enough audio (0.5 seconds)
                if len(audio_buffer) >= int(self.sample_rate * 0.5):
                    # Transcribe the buffered audio
                    segments, _ = self.model.transcribe(
                        audio_buffer,
                        beam_size=5,
                        language="en",
                        vad_filter=True
                    )

                    # Get the transcribed text
                    for segment in segments:
                        yield segment.text.strip()

                    # Clear the buffer once processed
                    audio_buffer = np.array([], dtype=np.float32)

            # Process any remaining audio in the buffer
            if len(audio_buffer) > 0:
                segments, _ = self.model.transcribe(
                    audio_buffer,
                    beam_size=5,
                    language="en",
                    vad_filter=True
                )

                for segment in segments:
                    yield segment.text.strip()

        except Exception as e:
            logger.error(f"Error in stream transcription: {str(e)}")
            yield f"Error: {str(e)}"