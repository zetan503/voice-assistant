"""
Home Assistant integration manager.
Coordinates all Home Assistant related functionality including security monitoring.
"""

import logging
import asyncio
import re
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime
import os

from app.integrations.home_assistant.client import HomeAssistantClient, EntityState
from app.integrations.yolink.sensor import YoLinkSensorManager, YoLinkDevice
from app.integrations.camera.recognition import CameraManager, DetectedFace
from app.integrations.security.perimeter import PerimeterMonitor, SecurityEvent, SecurityZone
from app.integrations.security.announcements import AnnouncementManager, AnnouncementType

from app.models.text_to_speech import TextToSpeech
from app.utils.audio import AudioIO

from config.settings import HOME_ASSISTANT, BASE_DIR

logger = logging.getLogger(__name__)


class HomeAssistantManager:
    """Manager for Home Assistant and related integrations."""

    def __init__(
        self,
        tts: Optional[TextToSpeech] = None,
        audio: Optional[AudioIO] = None
    ):
        """
        Initialize the Home Assistant manager.

        Args:
            tts: Text-to-speech engine
            audio: Audio output system
        """
        # Store references to voice components
        self.tts = tts
        self.audio = audio

        # Create clients/managers
        self.hass_client = HomeAssistantClient()
        self.yolink_manager = YoLinkSensorManager()
        self.camera_manager = CameraManager()
        self.perimeter_monitor = PerimeterMonitor(
            hass_client=self.hass_client,
            yolink_manager=self.yolink_manager
        )
        self.announcement_manager = AnnouncementManager(
            tts=self.tts,
            audio=self.audio
        )

        # Integration state
        self.is_initialized = False
        self.is_active = False

        # Create data directory if it doesn't exist
        data_dir = os.path.join(BASE_DIR, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    async def initialize(self) -> bool:
        """
        Initialize all integrations.

        Returns:
            True if initialization is successful, False otherwise
        """
        logger.info("Initializing Home Assistant integration")

        try:
            # Connect to Home Assistant
            if not await self.hass_client.connect():
                logger.warning("Failed to connect to Home Assistant, some features may be limited")

            # Initialize YoLink integration
            await self.yolink_manager.initialize()

            # Initialize camera integration
            await self.camera_manager.initialize()

            # Initialize perimeter monitoring
            await self.perimeter_monitor.initialize()

            # Start announcement manager
            await self.announcement_manager.start()

            # Set up event handlers
            self._setup_event_handlers()

            self.is_initialized = True
            logger.info("Home Assistant integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing Home Assistant integration: {str(e)}")
            return False

    async def start(self) -> bool:
        """
        Start all integration components.

        Returns:
            True if started successfully, False otherwise
        """
        if not self.is_initialized:
            if not await self.initialize():
                return False

        logger.info("Starting Home Assistant integration")

        try:
            # Start perimeter monitoring
            await self.perimeter_monitor.start_monitoring()

            # Announce that system is active
            await self.announcement_manager.announce_general(
                "Home security system is now active."
            )

            self.is_active = True
            logger.info("Home Assistant integration started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting Home Assistant integration: {str(e)}")
            return False

    async def stop(self) -> bool:
        """
        Stop all integration components.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.is_active:
            return False

        logger.info("Stopping Home Assistant integration")

        try:
            # Stop perimeter monitoring
            await self.perimeter_monitor.stop_monitoring()

            # Announce that system is stopping
            await self.announcement_manager.announce_general(
                "Home security system is now deactivated."
            )

            self.is_active = False
            logger.info("Home Assistant integration stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping Home Assistant integration: {str(e)}")
            return False

    async def shutdown(self):
        """Shutdown all integration components."""
        logger.info("Shutting down Home Assistant integration")

        try:
            # Stop perimeter monitoring
            if self.perimeter_monitor:
                await self.perimeter_monitor.stop_monitoring()

            # Stop announcement manager
            if self.announcement_manager:
                await self.announcement_manager.stop()

            # Shutdown camera integration
            if self.camera_manager:
                await self.camera_manager.shutdown()

            # Shutdown YoLink integration
            if self.yolink_manager:
                await self.yolink_manager.shutdown()

            # Disconnect from Home Assistant
            if self.hass_client:
                await self.hass_client.disconnect()

            self.is_active = False
            self.is_initialized = False
            logger.info("Home Assistant integration shut down successfully")

        except Exception as e:
            logger.error(f"Error shutting down Home Assistant integration: {str(e)}")

    def _setup_event_handlers(self):
        """Set up event handlers between components."""
        # Connect security events to announcements
        self.perimeter_monitor.set_security_event_callback(self._handle_security_event)
        self.perimeter_monitor.set_zone_status_change_callback(self._handle_zone_change)

        # Connect camera events to announcements
        self.camera_manager.set_known_face_callback(self._handle_known_face)
        self.camera_manager.set_unknown_face_callback(self._handle_unknown_face)

    async def _handle_security_event(self, event: SecurityEvent):
        """
        Handle security events.

        Args:
            event: Security event to handle
        """
        logger.info(f"Handling security event: {event.event_type}")

        # Pass to announcement manager
        await self.announcement_manager.handle_security_event(event)

    async def _handle_zone_change(self, zone: SecurityZone):
        """
        Handle security zone changes.

        Args:
            zone: Security zone that changed
        """
        logger.info(f"Handling zone change: {zone.zone_id} -> {zone.status}")

        # Pass to announcement manager
        await self.announcement_manager.handle_zone_change(zone)

    async def _handle_known_face(self, face: DetectedFace, image_data: bytes):
        """
        Handle known face detection.

        Args:
            face: Detected face
            image_data: Image data
        """
        logger.info(f"Known face detected: {face.name}")

        # Check if this is the mail carrier
        is_mailman = self._is_mail_carrier(face.name)

        # Pass to announcement manager
        await self.announcement_manager.handle_face_detected(face, is_mailman)

    async def _handle_unknown_face(self, face: DetectedFace, image_data: bytes):
        """
        Handle unknown face detection.

        Args:
            face: Detected face
            image_data: Image data
        """
        logger.info("Unknown face detected")

        # Pass to announcement manager
        await self.announcement_manager.handle_face_detected(face)

    def _is_mail_carrier(self, name: str) -> bool:
        """
        Check if a name matches the mail carrier.

        Args:
            name: Name to check

        Returns:
            True if name matches mail carrier, False otherwise
        """
        mail_carrier_names = ["mail", "carrier", "post", "postal", "usps", "fedex", "ups", "dhl", "postman", "mailman"]
        name_lower = name.lower()
        return any(carrier in name_lower for carrier in mail_carrier_names)

    async def get_security_status(self) -> Dict[str, Any]:
        """
        Get current security status.

        Returns:
            Dictionary with security status information
        """
        zones_status = {}
        for zone_id, zone in self.perimeter_monitor.zones.items():
            zones_status[zone_id] = {
                "name": zone.zone_name,
                "status": zone.status,
                "last_event": zone.last_event.dict() if zone.last_event else None
            }

        return {
            "is_active": self.is_active,
            "zones": zones_status,
            "recent_events": [event.dict() for event in await self.perimeter_monitor.get_recent_events()]
        }

    async def get_camera_status(self) -> Dict[str, Any]:
        """
        Get camera status information.

        Returns:
            Dictionary with camera status information
        """
        return {
            "last_faces": [face.dict() for face in self.camera_manager.last_detected_faces]
        }

    async def make_announcement(self, message: str, priority: bool = False) -> bool:
        """
        Make a general announcement.

        Args:
            message: Message to announce
            priority: Whether this is a priority announcement

        Returns:
            True if successful, False otherwise
        """
        try:
            await self.announcement_manager.announce_general(message, priority)
            return True
        except Exception as e:
            logger.error(f"Error making announcement: {str(e)}")
            return False

    async def reset_security_zone(self, zone_id: str) -> bool:
        """
        Reset a security zone to secure status.

        Args:
            zone_id: Zone ID to reset

        Returns:
            True if successful, False otherwise
        """
        return await self.perimeter_monitor.reset_zone(zone_id)

    async def get_sensor_states(self) -> List[Dict[str, Any]]:
        """
        Get states of all sensors.

        Returns:
            List of sensor states
        """
        if not self.hass_client.is_connected:
            await self.hass_client.connect()

        states = await self.hass_client.get_states(force_refresh=True)

        # Filter for sensor entities
        sensor_states = []
        for state in states:
            if state.entity_id.startswith("sensor.") or state.entity_id.startswith("binary_sensor."):
                sensor_states.append(state.dict())

        return sensor_states