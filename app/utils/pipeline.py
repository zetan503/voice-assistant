"""
Async processing pipeline for voice assistant.
Provides functionality for handling streaming input and output concurrently.
"""
import asyncio
import logging
import numpy as np
from typing import AsyncGenerator, Callable, Dict, List, Optional, Union

from app.models.llm import LocalLLM
from app.models.speech_to_text import SpeechToText
from app.models.text_to_speech import TextToSpeech
from app.models.wake_word import WakeWordDetector
from app.utils.audio import AudioIO
from app.utils.sound_effects import StarTrekSounds

logger = logging.getLogger(__name__)

class AsyncVoicePipeline:
    """
    Asynchronous pipeline for voice processing.
    Manages concurrent speech input/output and model interactions.
    """

    def __init__(self):
        """Initialize the voice processing pipeline."""
        # Initialize component models
        self.audio = AudioIO()
        self.wake_word = WakeWordDetector()
        self.stt = SpeechToText()
        self.llm = LocalLLM()
        self.tts = TextToSpeech()

        # Star Trek sound effects
        self.sound_effects = StarTrekSounds(sample_rate=self.audio.sample_rate)

        # State management
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.is_active = False

        # Buffer for collecting audio
        self.audio_buffer = np.array([], dtype=np.float32)

        # Lock for audio operations
        self.audio_lock = asyncio.Lock()

        # Events for synchronization
        self.wake_word_detected = asyncio.Event()
        self.speech_detected = asyncio.Event()
        self.response_ready = asyncio.Event()

        # Callbacks
        self.on_wake_word = None
        self.on_transcription = None
        self.on_response = None
        self.on_error = None

        # Sound effect settings
        self.use_sound_effects = True

    async def start(self):
        """Start the voice assistant pipeline."""
        self.is_active = True

        # Start the main tasks
        await asyncio.gather(
            self.wake_word_detection_loop(),
            self.conversation_loop()
        )

    async def stop(self):
        """Stop the voice assistant pipeline."""
        self.is_active = False
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False

        # Stop audio processing
        self.audio.stop_recording()
        self.audio.stop_playback()

    async def wake_word_detection_loop(self):
        """Run the wake word detection loop."""
        logger.info("Starting wake word detection loop")

        async def process_audio_chunk(chunk):
            # Check for wake word
            if self.wake_word.detect(chunk):
                logger.info("Wake word detected!")
                self.wake_word_detected.set()

                # Play acknowledgment sound
                if self.use_sound_effects:
                    acknowledgment_sound = self.sound_effects.generate_acknowledgment()
                    await self.sound_effects.play(acknowledgment_sound)

                if self.on_wake_word:
                    await self.on_wake_word()

        try:
            while self.is_active:
                # Reset the event
                self.wake_word_detected.clear()

                # Only detect wake words when not already in a conversation
                if not self.is_listening and not self.is_speaking:
                    # Start listening for wake word
                    async for chunk in self.audio.record_stream():
                        # Process the chunk for wake word detection
                        await process_audio_chunk(chunk)

                        # If wake word detected, break out of the recording loop
                        if self.wake_word_detected.is_set():
                            break

                        # Cooperative multitasking
                        await asyncio.sleep(0.01)

                # Wait a bit before checking again if we're not in detection mode
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in wake word detection loop: {str(e)}")
            if self.on_error:
                await self.on_error(f"Wake word error: {str(e)}")

    async def conversation_loop(self):
        """Run the conversation loop triggered by wake word."""
        logger.info("Starting conversation loop")

        try:
            while self.is_active:
                # Wait for wake word detection
                await self.wake_word_detected.wait()

                # Start conversation flow
                await self.handle_conversation()

                # Reset for next conversation
                self.wake_word_detected.clear()

        except Exception as e:
            logger.error(f"Error in conversation loop: {str(e)}")
            if self.on_error:
                await self.on_error(f"Conversation error: {str(e)}")

    async def handle_conversation(self):
        """Handle the complete conversation flow."""
        try:
            # Start listening for speech
            self.is_listening = True

            # Capture user speech
            user_audio = await self.capture_user_speech()

            if len(user_audio) > 0:
                # Speech detected, process it
                self.is_processing = True

                # Transcribe speech to text
                user_text = self.stt.transcribe(user_audio)
                logger.info(f"Transcription: {user_text}")

                if self.on_transcription:
                    await self.on_transcription(user_text)

                if user_text:
                    # Generate LLM response
                    llm_response = await self.generate_response(user_text)
                    logger.info(f"LLM response: {llm_response}")

                    if self.on_response:
                        await self.on_response(llm_response)

                    # Convert response to speech and play it
                    await self.speak_response(llm_response)

                self.is_processing = False

            # End the conversation
            self.is_listening = False

        except Exception as e:
            logger.error(f"Error handling conversation: {str(e)}")
            self.is_listening = False
            self.is_processing = False
            if self.on_error:
                await self.on_error(f"Conversation handling error: {str(e)}")

    async def capture_user_speech(self) -> np.ndarray:
        """
        Capture user speech until silence is detected.

        Returns:
            NumPy array of audio samples
        """
        logger.info("Capturing user speech...")
        self.audio_buffer = np.array([], dtype=np.float32)
        silence_counter = 0
        SILENCE_THRESHOLD = 0.02  # Amplitude threshold for silence
        MAX_SILENCE_CHUNKS = 30   # About 3 seconds of silence

        async for chunk in self.audio.record_stream():
            # Add the chunk to our buffer
            self.audio_buffer = np.append(self.audio_buffer, chunk)

            # Check if the chunk is mostly silence
            if np.mean(np.abs(chunk)) < SILENCE_THRESHOLD:
                silence_counter += 1
            else:
                silence_counter = 0

            # If we've had enough silent chunks, stop recording
            if silence_counter >= MAX_SILENCE_CHUNKS and len(self.audio_buffer) > self.audio.sample_rate:  # At least 1 second of audio
                break

            # Timeout after 10 seconds to prevent infinite recording
            if len(self.audio_buffer) > 10 * self.audio.sample_rate:
                break

        logger.info(f"Captured {len(self.audio_buffer)/self.audio.sample_rate:.2f} seconds of audio")
        return self.audio_buffer

    async def generate_response(self, text: str) -> str:
        """
        Generate a response using the LLM in Enterprise computer style.
        Processes user input for Star Trek-themed interactions.

        Args:
            text: User's transcribed speech

        Returns:
            Generated text response in Enterprise computer style
        """
        logger.info("Generating response in Enterprise computer style...")

        try:
            # Play computer processing sound
            if self.use_sound_effects:
                working_sound = self.sound_effects.generate_working()
                await self.sound_effects.play(working_sound)
                logger.info("Played computer processing sound effect")

            # Parse the user command to add Star Trek context if needed
            processed_text = self._process_user_command(text)

            # Generate response using the enhanced LLM
            response = self.llm.generate(processed_text)

            # Log the styled response
            logger.info(f"Enterprise computer response: {response}")

            # Play completion sound after generating response
            if self.use_sound_effects:
                completion_sound = self.sound_effects.generate_completion()
                await self.sound_effects.play(completion_sound)
                logger.info("Played completion sound effect")

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

            # Play error sound
            if self.use_sound_effects:
                error_sound = self.sound_effects.generate_error()
                await self.sound_effects.play(error_sound)
                logger.info("Played error sound effect")

            return "Unable to comply. System malfunction detected."

    def _process_user_command(self, text: str) -> str:
        """
        Process user command to enhance Star Trek interaction experience.
        Adds context or reformats questions to improve computer responses.

        Args:
            text: Original user speech text

        Returns:
            Processed command optimized for Enterprise computer responses
        """
        # Convert text to lowercase for easier pattern matching
        text_lower = text.lower()

        # Handle specific Star Trek command patterns
        if "status report" in text_lower:
            return "Provide a brief status report on all major ship systems"
        elif "set course" in text_lower or "plot course" in text_lower:
            return f"I need navigational assistance: {text}"
        elif "shields" in text_lower and ("up" in text_lower or "raise" in text_lower):
            return "Activate defensive systems and shield protocols"
        elif "red alert" in text_lower:
            return "Initiate red alert protocols and prepare ship systems for potential combat"
        elif "tea" in text_lower and "earl grey" in text_lower and "hot" in text_lower:
            return "Captain Picard's favorite tea order detected. Respond with a reference to the replicator"

        # Add context to general knowledge questions
        if text_lower.startswith("what is") or text_lower.startswith("who is") or text_lower.startswith("where is"):
            return f"As the Enterprise computer, answer this query: {text}"

        # For general commands, just pass through
        return text

    async def speak_response(self, text: str) -> None:
        """
        Convert text to speech and play it with interruption capability.
        Monitors for wake word while speaking to enable mid-response interruptions.

        Args:
            text: Text to speak
        """
        logger.info("Speaking response...")

        try:
            # Set speaking state
            self.is_speaking = True

            # Set up a task to monitor for interruptions
            interruption_detected = asyncio.Event()

            # Create a background task to monitor for wake word while speaking
            async def monitor_for_interruption():
                logger.debug("Starting interruption monitor")
                async for chunk in self.audio.record_stream():
                    # Check if wake word is detected during response
                    if self.wake_word.detect(chunk):
                        logger.info("Voice command interruption detected!")
                        interruption_detected.set()
                        break

                    # Stop monitoring if we're no longer speaking
                    if not self.is_speaking:
                        break

                    # Allow for cooperative multitasking
                    await asyncio.sleep(0.01)

                logger.debug("Interruption monitor stopped")

            # Start the interruption monitor
            interruption_monitor = asyncio.create_task(monitor_for_interruption())

            # Synthesize speech
            async for audio_chunk in self.tts.synthesize_stream(text):
                # Check for interruption before playing each chunk
                if interruption_detected.is_set():
                    logger.info("Response interrupted. Stopping speech output.")
                    break

                # Play the chunk
                await self.audio.play(audio_chunk)

                # Allow for cooperative multitasking
                await asyncio.sleep(0.01)

            # Clean up
            self.is_speaking = False

            # Cancel the interruption monitor if it's still running
            if not interruption_monitor.done():
                interruption_monitor.cancel()

            # If interrupted, immediately set the wake word detected event to trigger a new conversation
            if interruption_detected.is_set():
                self.wake_word_detected.set()
                # Play interruption acknowledgment sound
                if self.use_sound_effects:
                    interruption_sound = self.sound_effects.generate_interruption()
                    await self.sound_effects.play(interruption_sound)
                logger.info("Played interruption acknowledgment sound")

        except Exception as e:
            logger.error(f"Error speaking response: {str(e)}")
            self.is_speaking = False

    async def process_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None]
    ) -> AsyncGenerator[str, None]:
        """
        Process streaming audio and yield real-time transcriptions.

        Args:
            audio_stream: Generator yielding audio chunks

        Yields:
            Real-time transcriptions
        """
        # Process streaming transcription
        async for transcription in self.stt.transcribe_stream(audio_stream):
            yield transcription

            # Allow for cooperative multitasking
            await asyncio.sleep(0.01)

    def set_on_wake_word(self, callback: Callable[[], None]) -> None:
        """Set the callback for wake word detection."""
        self.on_wake_word = callback

    def set_on_transcription(self, callback: Callable[[str], None]) -> None:
        """Set the callback for speech transcription."""
        self.on_transcription = callback

    def set_on_response(self, callback: Callable[[str], None]) -> None:
        """Set the callback for LLM response."""
        self.on_response = callback

    def set_on_error(self, callback: Callable[[str], None]) -> None:
        """Set the callback for error handling."""
        self.on_error = callback