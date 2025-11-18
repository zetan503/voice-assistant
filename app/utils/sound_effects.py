"""
Sound effects module for Star Trek-style audio feedback.
Provides functions to generate and play Star Trek computer sound effects.
"""
import logging
import asyncio
import numpy as np
from typing import Optional, Callable
import sounddevice as sd

logger = logging.getLogger(__name__)

class StarTrekSounds:
    """
    Star Trek computer sound effects for voice assistant feedback.
    Generates synthesized sounds mimicking the Enterprise computer.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the sound effects generator.

        Args:
            sample_rate: Sample rate for generated audio
        """
        self.sample_rate = sample_rate

    def _generate_sine(
        self,
        frequency: float,
        duration: float,
        amplitude: float = 0.5,
        fade: bool = True
    ) -> np.ndarray:
        """
        Generate a sine wave tone.

        Args:
            frequency: Tone frequency in Hz
            duration: Duration in seconds
            amplitude: Amplitude of the tone (0.0 to 1.0)
            fade: Apply fade in/out if True

        Returns:
            NumPy array of audio samples
        """
        # Generate time array
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        # Generate sine wave
        tone = amplitude * np.sin(2 * np.pi * frequency * t)

        # Apply fade in/out if requested
        if fade:
            fade_len = min(int(0.05 * self.sample_rate), len(tone) // 4)
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)

            tone[:fade_len] *= fade_in
            tone[-fade_len:] *= fade_out

        return tone

    def _generate_sweep(
        self,
        start_freq: float,
        end_freq: float,
        duration: float,
        amplitude: float = 0.5
    ) -> np.ndarray:
        """
        Generate a frequency sweep (chirp).

        Args:
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz
            duration: Duration in seconds
            amplitude: Amplitude of the sweep (0.0 to 1.0)

        Returns:
            NumPy array of audio samples
        """
        # Generate time array
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        # Calculate frequency at each time step (logarithmic sweep)
        freq = start_freq * (end_freq / start_freq) ** (t / duration)

        # Generate sweep
        phi = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        sweep = amplitude * np.sin(phi)

        # Apply fade in/out
        fade_len = min(int(0.01 * self.sample_rate), len(sweep) // 4)
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)

        sweep[:fade_len] *= fade_in
        sweep[-fade_len:] *= fade_out

        return sweep

    def generate_acknowledgment(self) -> np.ndarray:
        """
        Generate the classic Star Trek computer acknowledgment sound.
        A short, high-pitched beep indicating the computer is ready.

        Returns:
            NumPy array of audio samples
        """
        # Short high beep followed by shorter higher beep
        beep1 = self._generate_sine(1800, 0.15, 0.4)
        beep2 = self._generate_sine(2100, 0.05, 0.3)

        # Small gap between beeps
        gap = np.zeros(int(0.02 * self.sample_rate))

        # Combine beeps
        return np.concatenate([beep1, gap, beep2])

    def generate_working(self) -> np.ndarray:
        """
        Generate a "working" or "processing" sound effect.
        A rhythmic pattern suggesting the computer is thinking.

        Returns:
            NumPy array of audio samples
        """
        # Series of ascending beeps
        duration = 0.1
        gap_duration = 0.05
        freqs = [1400, 1600, 1800, 2000]

        # Create tones and gaps
        tones = [self._generate_sine(freq, duration, 0.3) for freq in freqs]
        gap = np.zeros(int(gap_duration * self.sample_rate))

        # Combine with gaps between tones
        result = np.array([], dtype=np.float32)
        for tone in tones:
            result = np.concatenate([result, tone, gap])

        return result

    def generate_alert(self) -> np.ndarray:
        """
        Generate an alert or warning sound.
        Simulates the Enterprise alert notification.

        Returns:
            NumPy array of audio samples
        """
        # Alternating lower and higher tones
        low_freq = 800
        high_freq = 1200
        duration = 0.2
        repeat = 3

        # Create the tones
        low_tone = self._generate_sine(low_freq, duration, 0.5)
        high_tone = self._generate_sine(high_freq, duration, 0.5)

        # Combine with alternation
        result = np.array([], dtype=np.float32)
        for _ in range(repeat):
            result = np.concatenate([result, low_tone, high_tone])

        return result

    def generate_completion(self) -> np.ndarray:
        """
        Generate a task completion sound.
        A descending tone indicating the computer has completed a task.

        Returns:
            NumPy array of audio samples
        """
        # Descending sweep
        return self._generate_sweep(2200, 1600, 0.3, 0.4)

    def generate_error(self) -> np.ndarray:
        """
        Generate an error or malfunction sound.
        A harsh, dissonant tone suggesting something went wrong.

        Returns:
            NumPy array of audio samples
        """
        # Harsh, dissonant tone with modulation
        duration = 0.4
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        # Generate a complex, dissonant sound
        tone1 = 0.3 * np.sin(2 * np.pi * 900 * t)
        tone2 = 0.3 * np.sin(2 * np.pi * 940 * t)  # Dissonant with 900Hz

        # Apply amplitude modulation
        modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 8 * t)  # 8Hz modulation

        # Combine tones and apply modulation
        combined = (tone1 + tone2) * modulator

        # Apply fade out
        fade_len = min(int(0.05 * self.sample_rate), len(combined) // 4)
        fade_out = np.linspace(1, 0, fade_len)
        combined[-fade_len:] *= fade_out

        return combined

    def generate_interruption(self) -> np.ndarray:
        """
        Generate an interruption acknowledgment sound.
        A quick sound indicating the computer has been interrupted.

        Returns:
            NumPy array of audio samples
        """
        # Quick descending sweep with sharp cutoff
        sweep = self._generate_sweep(2000, 1000, 0.15, 0.4)
        cutoff = int(len(sweep) * 0.8)  # Sharp cutoff
        return sweep[:cutoff]

    async def play(self, audio: np.ndarray) -> None:
        """
        Play audio asynchronously.

        Args:
            audio: NumPy array of audio samples
        """
        try:
            # Play the sound
            sd.play(audio, self.sample_rate)
            # Wait for the sound to finish
            await asyncio.sleep(len(audio) / self.sample_rate)
            # Stop the sound
            sd.stop()

        except Exception as e:
            logger.error(f"Error playing sound effect: {str(e)}")