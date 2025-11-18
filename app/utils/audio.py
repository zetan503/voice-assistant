"""
Audio utilities for handling audio input and output in voice assistant.
Provides functionality for capturing and playing audio streams.
"""
import logging
import asyncio
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import AsyncGenerator, Dict, Optional, Union, Callable

from config.settings import AUDIO

logger = logging.getLogger(__name__)

class AudioIO:
    """
    Audio input/output handler.
    Provides methods for capturing and playing audio streams.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize audio I/O with optional custom configuration.

        Args:
            config: Optional custom audio configuration
        """
        self.config = config or AUDIO
        self.sample_rate = self.config["sample_rate"]
        self.channels = self.config["channels"]
        self.chunk_size = self.config["chunk_size"]
        self.device_index = self.config["device_index"]
        self.stream = None
        self.is_recording = False
        self.is_playing = False

    async def record(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Record audio for a specified duration.

        Args:
            duration: Duration in seconds, or None for indefinite

        Returns:
            NumPy array of recorded audio samples
        """
        frames = []

        def callback(indata, frame_count, time_info, status):
            if status:
                logger.warning(f"Audio recording status: {status}")
            frames.append(indata.copy())

        try:
            # Set up the stream
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback,
                blocksize=self.chunk_size,
                device=self.device_index
            )

            with stream:
                if duration is not None:
                    await asyncio.sleep(duration)
                else:
                    # Indefinite recording until is_recording is False
                    self.is_recording = True
                    while self.is_recording:
                        await asyncio.sleep(0.1)

            # Combine all frames
            return np.concatenate(frames, axis=0) if frames else np.array([], dtype=np.float32)

        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            return np.array([], dtype=np.float32)

    async def record_stream(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream audio from microphone asynchronously.

        Yields:
            Audio chunks as NumPy arrays
        """
        queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            if status:
                logger.warning(f"Audio streaming status: {status}")
            # Add the audio chunk to the queue
            queue.put_nowait(indata.copy().squeeze())

        try:
            # Set up the stream
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback,
                blocksize=self.chunk_size,
                device=self.device_index
            )

            with stream:
                self.is_recording = True
                while self.is_recording:
                    # Yield audio chunks as they become available
                    chunk = await queue.get()
                    yield chunk

        except Exception as e:
            logger.error(f"Error in audio stream: {str(e)}")

        finally:
            self.is_recording = False
            logger.debug("Audio streaming stopped")

    async def play(self, audio: np.ndarray) -> None:
        """
        Play audio synchronously.

        Args:
            audio: NumPy array of audio samples
        """
        try:
            self.is_playing = True
            sd.play(audio, self.sample_rate)
            sd.wait()
            self.is_playing = False

        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
            self.is_playing = False

    async def play_stream(self, audio_stream: AsyncGenerator[np.ndarray, None]) -> None:
        """
        Play streaming audio asynchronously.

        Args:
            audio_stream: Generator yielding audio chunks
        """
        try:
            # Set up the output stream
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                device=self.device_index
            )

            with stream:
                self.is_playing = True
                async for chunk in audio_stream:
                    if not self.is_playing:
                        break
                    # Play the chunk
                    stream.write(chunk)
                    # Small delay to prevent buffer overrun
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in audio playback stream: {str(e)}")

        finally:
            self.is_playing = False
            logger.debug("Audio playback stopped")

    def stop_recording(self) -> None:
        """Stop any ongoing recording."""
        self.is_recording = False

    def stop_playback(self) -> None:
        """Stop any ongoing playback."""
        self.is_playing = False
        sd.stop()

    async def monitor_with_callback(
        self,
        callback: Callable[[np.ndarray], None],
        chunk_size: Optional[int] = None
    ) -> None:
        """
        Monitor audio input and call the callback for each chunk.
        Useful for wake word detection or continuous speech recognition.

        Args:
            callback: Function to call with each audio chunk
            chunk_size: Optional custom chunk size
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        async for chunk in self.record_stream():
            # Process the chunk with the callback
            callback(chunk)

            # Allow for cooperative multitasking
            await asyncio.sleep(0.001)

    def save_to_file(self, audio: np.ndarray, file_path: str) -> None:
        """
        Save audio data to a file.

        Args:
            audio: NumPy array of audio samples
            file_path: Path to save the audio file
        """
        try:
            sf.write(file_path, audio, self.sample_rate)
            logger.debug(f"Audio saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving audio to file: {str(e)}")