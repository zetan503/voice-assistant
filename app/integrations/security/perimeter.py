"""
Security perimeter monitoring module.
Provides functionality for monitoring security sensors and handling alerts.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime, timedelta
from pydantic import BaseModel

from config.settings import SECURITY
from app.integrations.home_assistant.client import HomeAssistantClient, EntityState
from app.integrations.yolink.sensor import YoLinkSensorManager, YoLinkDevice

logger = logging.getLogger(__name__)


class SecurityEvent(BaseModel):
    """Model representing a security event."""
    event_id: str
    event_type: str
    source: str
    source_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class SecurityZone(BaseModel):
    """Model representing a security zone."""
    zone_id: str
    zone_name: str
    sensors: List[str]
    status: str = "secure"  # secure, alert, disabled
    last_event: Optional[SecurityEvent] = None


class PerimeterMonitor:
    """Manager for security perimeter monitoring."""

    def __init__(
        self,
        hass_client: Optional[HomeAssistantClient] = None,
        yolink_manager: Optional[YoLinkSensorManager] = None,
        perimeter_sensors: List[str] = None,
        alert_cooldown: int = None,
        monitoring_interval: int = None,
    ):
        """
        Initialize the perimeter monitor.

        Args:
            hass_client: Home Assistant client
            yolink_manager: YoLink sensor manager
            perimeter_sensors: List of perimeter sensor entity IDs
            alert_cooldown: Cooldown period for alerts in seconds
            monitoring_interval: Monitoring interval in seconds
        """
        self.hass_client = hass_client
        self.yolink_manager = yolink_manager
        self.perimeter_sensors = perimeter_sensors or SECURITY.get("perimeter_sensors")
        self.alert_cooldown = alert_cooldown or SECURITY.get("alert_cooldown")
        self.monitoring_interval = monitoring_interval or SECURITY.get("monitoring_interval")

        # Security zones
        self.zones: Dict[str, SecurityZone] = {
            "perimeter": SecurityZone(
                zone_id="perimeter",
                zone_name="Perimeter",
                sensors=self.perimeter_sensors,
                status="secure"
            )
        }

        # Event history
        self.events: List[SecurityEvent] = []
        self._max_events = 100

        # Monitoring state
        self.is_monitoring = False
        self._monitoring_task = None
        self._last_alert_time = 0

        # Event callbacks
        self.on_security_event = None
        self.on_zone_status_change = None

    async def initialize(self) -> bool:
        """
        Initialize the perimeter monitor.

        Returns:
            True if initialization is successful, False otherwise
        """
        # Initialize Home Assistant connection if needed
        if self.hass_client and not self.hass_client.is_connected:
            if not await self.hass_client.connect():
                logger.warning("Failed to connect to Home Assistant, perimeter monitoring may be limited")

        # Set up YoLink motion callback if available
        if self.yolink_manager:
            self.yolink_manager.set_motion_callback(self._handle_yolink_motion)

        return True

    async def start_monitoring(self) -> bool:
        """
        Start perimeter monitoring.

        Returns:
            True if started, False if already running
        """
        if self.is_monitoring:
            return False

        self.is_monitoring = True
        logger.info("Starting perimeter monitoring")

        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        return True

    async def stop_monitoring(self) -> bool:
        """
        Stop perimeter monitoring.

        Returns:
            True if stopped, False if not running
        """
        if not self.is_monitoring:
            return False

        logger.info("Stopping perimeter monitoring")
        self.is_monitoring = False

        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        return True

    async def get_zone_status(self, zone_id: str) -> Optional[str]:
        """
        Get the status of a security zone.

        Args:
            zone_id: Zone ID to get status for

        Returns:
            Zone status or None if zone not found
        """
        if zone_id in self.zones:
            return self.zones[zone_id].status
        return None

    async def _monitoring_loop(self):
        """Monitor security zones."""
        logger.info("Starting security monitoring loop")
        try:
            while self.is_monitoring:
                # Check all sensors in all zones
                for zone_id, zone in self.zones.items():
                    old_status = zone.status
                    await self._check_zone_sensors(zone)

                    # If status changed, trigger callback
                    if old_status != zone.status and self.on_zone_status_change:
                        await self.on_zone_status_change(zone)

                # Wait for next monitoring interval
                await asyncio.sleep(self.monitoring_interval)

        except asyncio.CancelledError:
            logger.info("Security monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in security monitoring loop: {str(e)}")
            self.is_monitoring = False

    async def _check_zone_sensors(self, zone: SecurityZone):
        """
        Check all sensors in a zone.

        Args:
            zone: Zone to check
        """
        # Skip if no Home Assistant client
        if not self.hass_client or not self.hass_client.is_connected:
            return

        # Skip if zone is disabled
        if zone.status == "disabled":
            return

        triggered_sensors = []

        # Check each sensor
        for sensor_id in zone.sensors:
            if not sensor_id:
                continue

            sensor_state = await self.hass_client.get_state(sensor_id)
            if not sensor_state:
                continue

            # Check sensor state (depends on sensor type)
            is_triggered = False

            # For binary sensors, 'on' typically means triggered
            if sensor_state.state == "on":
                is_triggered = True

            # Some sensors use 'detected' or other values
            elif sensor_state.state in ["detected", "motion", "open", "unlocked"]:
                is_triggered = True

            # For numeric sensors, check against a threshold
            elif sensor_state.state.isdigit():
                # This would depend on the sensor type, just an example
                threshold = sensor_state.attributes.get("threshold", 50)
                if int(sensor_state.state) > threshold:
                    is_triggered = True

            if is_triggered:
                triggered_sensors.append(sensor_id)

        # Update zone status
        if triggered_sensors:
            # Check cooldown
            current_time = time.time()
            if current_time - self._last_alert_time >= self.alert_cooldown:
                self._last_alert_time = current_time

                # Create security event
                event = SecurityEvent(
                    event_id=f"event_{int(current_time)}",
                    event_type="perimeter_breach",
                    source="home_assistant",
                    source_id=triggered_sensors[0],  # Use first triggered sensor
                    timestamp=datetime.now(),
                    metadata={
                        "zone": zone.zone_id,
                        "triggered_sensors": triggered_sensors,
                    }
                )

                # Update zone
                zone.status = "alert"
                zone.last_event = event

                # Add to events history
                self.events.append(event)
                if len(self.events) > self._max_events:
                    self.events.pop(0)  # Remove oldest event

                # Trigger callback
                if self.on_security_event:
                    await self.on_security_event(event)

                logger.warning(f"Security alert in {zone.zone_name}: {triggered_sensors}")
        else:
            # Reset to secure if no sensors are triggered
            if zone.status == "alert":
                zone.status = "secure"
                logger.info(f"Zone {zone.zone_name} is now secure")

    async def _handle_yolink_motion(self, device: YoLinkDevice):
        """
        Handle motion detection from YoLink sensors.

        Args:
            device: YoLink device that detected motion
        """
        # Check cooldown
        current_time = time.time()
        if current_time - self._last_alert_time < self.alert_cooldown:
            return

        self._last_alert_time = current_time

        # Create security event
        event = SecurityEvent(
            event_id=f"event_{int(current_time)}",
            event_type="motion_detected",
            source="yolink",
            source_id=device.device_id,
            timestamp=datetime.now(),
            metadata={
                "device_name": device.device_name,
                "device_type": device.device_type,
                "state": device.state,
            }
        )

        # Update perimeter zone
        if "perimeter" in self.zones:
            self.zones["perimeter"].status = "alert"
            self.zones["perimeter"].last_event = event

            # Trigger zone change callback
            if self.on_zone_status_change:
                await self.on_zone_status_change(self.zones["perimeter"])

        # Add to events history
        self.events.append(event)
        if len(self.events) > self._max_events:
            self.events.pop(0)  # Remove oldest event

        # Trigger callback
        if self.on_security_event:
            await self.on_security_event(event)

        logger.warning(f"YoLink motion detected: {device.device_name}")

    def set_security_event_callback(self, callback: Callable[[SecurityEvent], Awaitable[None]]):
        """
        Set callback for security events.

        Args:
            callback: Async function to call when a security event occurs
        """
        self.on_security_event = callback

    def set_zone_status_change_callback(self, callback: Callable[[SecurityZone], Awaitable[None]]):
        """
        Set callback for zone status changes.

        Args:
            callback: Async function to call when a zone status changes
        """
        self.on_zone_status_change = callback

    async def get_recent_events(self, limit: int = 10) -> List[SecurityEvent]:
        """
        Get recent security events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent security events
        """
        return self.events[-limit:]

    async def reset_zone(self, zone_id: str) -> bool:
        """
        Reset a zone to secure status.

        Args:
            zone_id: Zone ID to reset

        Returns:
            True if successful, False otherwise
        """
        if zone_id in self.zones:
            self.zones[zone_id].status = "secure"
            logger.info(f"Zone {self.zones[zone_id].zone_name} manually reset to secure")

            # Trigger zone change callback
            if self.on_zone_status_change:
                await self.on_zone_status_change(self.zones[zone_id])

            return True
        return False