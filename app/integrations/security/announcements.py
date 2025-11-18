"""
Security announcement system.
Provides functionality for automated announcements based on security events.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum

from app.models.text_to_speech import TextToSpeech
from app.utils.audio import AudioIO
from app.integrations.camera.recognition import DetectedFace
from app.integrations.security.perimeter import SecurityEvent, SecurityZone

logger = logging.getLogger(__name__)


class AnnouncementType(str, Enum):
    """Types of security announcements."""
    MOTION = "motion"
    PERIMETER = "perimeter"
    MAILMAN = "mailman"
    KNOWN_VISITOR = "known_visitor"
    UNKNOWN_VISITOR = "unknown_visitor"
    GENERAL = "general"


class AnnouncementManager:
    """Manager for automated security announcements."""

    def __init__(self, tts: TextToSpeech = None, audio: AudioIO = None):
        """
        Initialize the announcement manager.

        Args:
            tts: Text-to-speech engine
            audio: Audio output manager
        """
        self.tts = tts
        self.audio = audio

        # Cooldown management
        self._announcement_cooldowns: Dict[str, float] = {}
        self._default_cooldown = 60  # seconds

        # Custom announcement templates
        self.announcement_templates = {
            AnnouncementType.MOTION: "Motion detected in {location}.",
            AnnouncementType.PERIMETER: "Security alert. Perimeter breach detected in {location}.",
            AnnouncementType.MAILMAN: "The mail carrier has arrived.",
            AnnouncementType.KNOWN_VISITOR: "{name} is at the door.",
            AnnouncementType.UNKNOWN_VISITOR: "There is an unknown visitor at the door.",
            AnnouncementType.GENERAL: "{message}"
        }

        # Queue for announcements
        self._announcement_queue = asyncio.Queue()
        self._processing_task = None
        self.is_processing = False

    async def start(self):
        """Start the announcement processing loop."""
        if self._processing_task is None or self._processing_task.done():
            self.is_processing = True
            self._processing_task = asyncio.create_task(self._process_announcements())
            logger.info("Announcement manager started")

    async def stop(self):
        """Stop the announcement processing loop."""
        self.is_processing = False
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None
            logger.info("Announcement manager stopped")

    def set_cooldown(self, announcement_type: AnnouncementType, seconds: int):
        """
        Set cooldown for a specific announcement type.

        Args:
            announcement_type: Type of announcement
            seconds: Cooldown in seconds
        """
        self._announcement_cooldowns[announcement_type] = seconds

    def set_template(self, announcement_type: AnnouncementType, template: str):
        """
        Set template for a specific announcement type.

        Args:
            announcement_type: Type of announcement
            template: Announcement template string
        """
        self.announcement_templates[announcement_type] = template

    async def announce_motion(self, location: str, priority: bool = False):
        """
        Announce motion detection.

        Args:
            location: Location of motion
            priority: Whether to bypass cooldown
        """
        if self._check_cooldown(AnnouncementType.MOTION, priority):
            message = self.announcement_templates[AnnouncementType.MOTION].format(location=location)
            await self.queue_announcement(message, AnnouncementType.MOTION)

    async def announce_perimeter_breach(self, location: str, priority: bool = False):
        """
        Announce perimeter breach.

        Args:
            location: Location of breach
            priority: Whether to bypass cooldown
        """
        if self._check_cooldown(AnnouncementType.PERIMETER, priority):
            message = self.announcement_templates[AnnouncementType.PERIMETER].format(location=location)
            await self.queue_announcement(message, AnnouncementType.PERIMETER, priority=True)

    async def announce_mailman(self, priority: bool = False):
        """
        Announce mail carrier arrival.

        Args:
            priority: Whether to bypass cooldown
        """
        if self._check_cooldown(AnnouncementType.MAILMAN, priority):
            message = self.announcement_templates[AnnouncementType.MAILMAN]
            await self.queue_announcement(message, AnnouncementType.MAILMAN)

    async def announce_known_visitor(self, name: str, priority: bool = False):
        """
        Announce known visitor.

        Args:
            name: Name of visitor
            priority: Whether to bypass cooldown
        """
        if self._check_cooldown(AnnouncementType.KNOWN_VISITOR, priority):
            message = self.announcement_templates[AnnouncementType.KNOWN_VISITOR].format(name=name)
            await self.queue_announcement(message, AnnouncementType.KNOWN_VISITOR)

    async def announce_unknown_visitor(self, priority: bool = False):
        """
        Announce unknown visitor.

        Args:
            priority: Whether to bypass cooldown
        """
        if self._check_cooldown(AnnouncementType.UNKNOWN_VISITOR, priority):
            message = self.announcement_templates[AnnouncementType.UNKNOWN_VISITOR]
            await self.queue_announcement(message, AnnouncementType.UNKNOWN_VISITOR)

    async def announce_general(self, message: str, priority: bool = False):
        """
        Make a general announcement.

        Args:
            message: Announcement message
            priority: Whether to bypass cooldown
        """
        if self._check_cooldown(AnnouncementType.GENERAL, priority):
            formatted = self.announcement_templates[AnnouncementType.GENERAL].format(message=message)
            await self.queue_announcement(formatted, AnnouncementType.GENERAL)

    async def queue_announcement(self, message: str, announcement_type: AnnouncementType, priority: bool = False):
        """
        Queue an announcement for processing.

        Args:
            message: Announcement message
            announcement_type: Type of announcement
            priority: Whether this is a priority announcement
        """
        await self._announcement_queue.put({
            "message": message,
            "type": announcement_type,
            "priority": priority,
            "timestamp": datetime.now()
        })
        logger.info(f"Queued announcement: {message}")

    async def _process_announcements(self):
        """Process queued announcements."""
        logger.info("Starting announcement processing loop")
        try:
            while self.is_processing:
                if self._announcement_queue.empty():
                    await asyncio.sleep(0.5)
                    continue

                # Get next announcement
                announcement = await self._announcement_queue.get()
                message = announcement["message"]

                logger.info(f"Processing announcement: {message}")

                # Play announcement
                await self._play_announcement(message)

                # Mark as done
                self._announcement_queue.task_done()

                # Update cooldown
                self._announcement_cooldowns[announcement["type"]] = time.time() + self._get_cooldown_time(announcement["type"])

                # Short delay before next announcement
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Announcement processing loop cancelled")
        except Exception as e:
            logger.error(f"Error in announcement processing loop: {str(e)}")
            self.is_processing = False

    async def _play_announcement(self, message: str):
        """
        Play an announcement through TTS and audio output.

        Args:
            message: Message to announce
        """
        try:
            if self.tts and self.audio:
                # Synthesize speech
                async for audio_chunk in self.tts.synthesize_stream(message):
                    # Play the audio
                    await self.audio.play(audio_chunk)
                    await asyncio.sleep(0.01)  # Cooperative multitasking
            else:
                logger.warning("Cannot play announcement: TTS or Audio not available")
                logger.info(f"Announcement text: {message}")
        except Exception as e:
            logger.error(f"Error playing announcement: {str(e)}")

    def _check_cooldown(self, announcement_type: AnnouncementType, bypass: bool = False) -> bool:
        """
        Check if an announcement type is in cooldown.

        Args:
            announcement_type: Type of announcement to check
            bypass: Whether to bypass cooldown check

        Returns:
            True if announcement can be made, False if in cooldown
        """
        if bypass:
            return True

        current_time = time.time()
        last_time = self._announcement_cooldowns.get(announcement_type, 0)
        cooldown_time = self._get_cooldown_time(announcement_type)

        return current_time - last_time >= cooldown_time

    def _get_cooldown_time(self, announcement_type: AnnouncementType) -> float:
        """
        Get cooldown time for an announcement type.

        Args:
            announcement_type: Type of announcement

        Returns:
            Cooldown time in seconds
        """
        # Default cooldown times for different announcement types
        default_times = {
            AnnouncementType.MOTION: 60,
            AnnouncementType.PERIMETER: 30,
            AnnouncementType.MAILMAN: 300,
            AnnouncementType.KNOWN_VISITOR: 60,
            AnnouncementType.UNKNOWN_VISITOR: 60,
            AnnouncementType.GENERAL: 10
        }

        return default_times.get(announcement_type, self._default_cooldown)

    async def handle_security_event(self, event: SecurityEvent):
        """
        Handle security event by making appropriate announcement.

        Args:
            event: Security event to handle
        """
        if event.event_type == "perimeter_breach":
            location = event.metadata.get("zone", "unknown area")
            await self.announce_perimeter_breach(location)
        elif event.event_type == "motion_detected":
            location = event.metadata.get("device_name", "unknown area")
            await self.announce_motion(location)

    async def handle_face_detected(self, face: DetectedFace, is_mailman: bool = False):
        """
        Handle face detection by making appropriate announcement.

        Args:
            face: Detected face
            is_mailman: Whether the face is identified as a mail carrier
        """
        if is_mailman:
            await self.announce_mailman()
        elif face.name:
            await self.announce_known_visitor(face.name)
        else:
            await self.announce_unknown_visitor()

    async def handle_zone_change(self, zone: SecurityZone):
        """
        Handle zone status change by making appropriate announcement.

        Args:
            zone: Security zone that changed status
        """
        if zone.status == "alert":
            await self.announce_perimeter_breach(zone.zone_name)
        elif zone.status == "secure" and zone.last_event and zone.last_event.event_type == "perimeter_breach":
            # Announce all-clear if zone was previously in alert
            await self.announce_general(f"All clear in {zone.zone_name}. The area is now secure.")

    async def check_queue_size(self) -> int:
        """
        Get the current size of the announcement queue.

        Returns:
            Number of pending announcements
        """
        return self._announcement_queue.qsize()