"""
YoLink motion sensor integration module.
Provides functionality for connecting to YoLink sensors and handling motion events.
"""

import logging
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime, timedelta
from pydantic import BaseModel

from config.settings import YOLINK

logger = logging.getLogger(__name__)


class YoLinkDevice(BaseModel):
    """Model representing a YoLink device."""
    device_id: str
    device_name: str
    device_type: str
    state: Dict[str, Any] = {}
    last_update: Optional[datetime] = None


class YoLinkSensorManager:
    """Manager for YoLink motion sensors."""

    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        redirect_uri: str = None,
        user_access_token: str = None,
        device_list: List[str] = None
    ):
        """
        Initialize the YoLink sensor manager.

        Args:
            client_id: YoLink API client ID
            client_secret: YoLink API client secret
            redirect_uri: OAuth redirect URI
            user_access_token: User access token (if already authorized)
            device_list: List of device IDs to monitor
        """
        self.client_id = client_id or YOLINK.get("client_id")
        self.client_secret = client_secret or YOLINK.get("client_secret")
        self.redirect_uri = redirect_uri or YOLINK.get("redirect_uri")
        self.user_access_token = user_access_token or YOLINK.get("user_access_token")
        self.device_list = device_list or YOLINK.get("device_list")

        if not self.client_id or not self.client_secret:
            logger.warning("YoLink client ID or client secret not provided. Integration will not work.")

        self.api_url = "https://api.yosmart.com/open/yolink"
        self.api_version = "v2"
        self.devices: Dict[str, YoLinkDevice] = {}
        self.session = None

        # Event callback
        self.motion_detected_callback = None
        self._event_task = None
        self._polling_interval = 5  # seconds

        # Auth token
        self.access_token = None
        self.token_expires_at = 0

    async def initialize(self) -> bool:
        """
        Initialize the YoLink integration.

        Returns:
            True if initialization is successful, False otherwise
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()

        # Get access token if we don't have one or if it's expired
        if not self.access_token or time.time() >= self.token_expires_at:
            if not await self._get_access_token():
                return False

        # Get devices
        await self._get_devices()

        # Start monitoring for events
        if self._event_task is None or self._event_task.done():
            self._event_task = asyncio.create_task(self._monitor_devices())

        return bool(self.devices)

    async def shutdown(self):
        """Shutdown the YoLink integration."""
        if self._event_task and not self._event_task.done():
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
            self._event_task = None

        if self.session:
            await self.session.close()
            self.session = None

    async def _get_access_token(self) -> bool:
        """
        Get access token for YoLink API.

        Returns:
            True if token obtained successfully, False otherwise
        """
        if self.user_access_token:
            # Use pre-provided user access token
            self.access_token = self.user_access_token
            self.token_expires_at = time.time() + 86400  # Assume it's valid for a day
            return True

        try:
            # Exchange client credentials for access token
            async with self.session.post(
                "https://api.yosmart.com/open/token/get",
                json={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.access_token = data.get("access_token")
                    # Calculate expiry time (token valid for data.get("expires_in") seconds)
                    self.token_expires_at = time.time() + int(data.get("expires_in", 3600))
                    logger.info("YoLink access token obtained successfully")
                    return True
                else:
                    logger.error(f"Failed to get YoLink access token. Status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error getting YoLink access token: {str(e)}")
            return False

    async def _api_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the YoLink API.

        Args:
            method: API method to call
            params: Parameters for the method

        Returns:
            Response data or empty dict on error
        """
        if not self.access_token or time.time() >= self.token_expires_at:
            if not await self._get_access_token():
                return {}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

        payload = {
            "method": method,
            "time": int(time.time()),
        }

        if params:
            payload["params"] = params

        try:
            async with self.session.post(
                f"{self.api_url}/{self.api_version}",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == "000000":  # Success code
                        return data.get("data", {})
                    else:
                        logger.error(f"YoLink API error: {data.get('code')} - {data.get('desc')}")
                        return {}
                else:
                    logger.error(f"YoLink API request failed. Status: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error making YoLink API request: {str(e)}")
            return {}

    async def _get_devices(self) -> List[YoLinkDevice]:
        """
        Get list of YoLink devices.

        Returns:
            List of YoLink devices
        """
        devices_data = await self._api_request("Home.getDeviceList")
        device_list = devices_data.get("devices", [])

        # Update devices dict
        for device_info in device_list:
            device_id = device_info.get("deviceId")
            if not device_id:
                continue

            # Skip if not in our device list (if provided)
            if self.device_list and device_id not in self.device_list:
                continue

            # Get device details
            device_data = await self._api_request("Device.getDeviceInfo", {"deviceId": device_id})
            if not device_data:
                continue

            device = YoLinkDevice(
                device_id=device_id,
                device_name=device_info.get("name", "Unknown Device"),
                device_type=device_info.get("type", "Unknown"),
                state=device_data.get("state", {}),
                last_update=datetime.now()
            )

            self.devices[device_id] = device
            logger.info(f"Added YoLink device: {device.device_name} ({device.device_type})")

        return list(self.devices.values())

    async def get_device_state(self, device_id: str) -> Dict[str, Any]:
        """
        Get current state of a device.

        Args:
            device_id: Device ID to get state for

        Returns:
            Device state or empty dict if device not found
        """
        if device_id not in self.devices:
            logger.warning(f"Device {device_id} not found")
            return {}

        # Get device details
        device_data = await self._api_request("Device.getState", {"deviceId": device_id})
        if not device_data:
            return {}

        # Update device state
        self.devices[device_id].state = device_data.get("state", {})
        self.devices[device_id].last_update = datetime.now()

        return self.devices[device_id].state

    async def _monitor_devices(self):
        """Monitor devices for state changes."""
        logger.info("Starting YoLink device monitoring")
        try:
            while True:
                for device_id in list(self.devices.keys()):
                    device = self.devices[device_id]
                    if device.device_type.lower() in ["motion_sensor", "motionsensor"]:
                        old_state = device.state.copy()
                        new_state = await self.get_device_state(device_id)

                        # Check if motion was detected
                        if new_state.get("state") == "alert" and old_state.get("state") != "alert":
                            logger.info(f"Motion detected by {device.device_name}")
                            if self.motion_detected_callback:
                                asyncio.create_task(self.motion_detected_callback(device))

                # Wait before next poll
                await asyncio.sleep(self._polling_interval)
        except asyncio.CancelledError:
            logger.info("YoLink device monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in YoLink device monitoring: {str(e)}")

    def set_motion_callback(self, callback: Callable[[YoLinkDevice], Awaitable[None]]):
        """
        Set callback for motion detection events.

        Args:
            callback: Async function to call when motion is detected
        """
        self.motion_detected_callback = callback

    async def get_motion_sensors(self) -> List[YoLinkDevice]:
        """
        Get list of motion sensors.

        Returns:
            List of motion sensor devices
        """
        return [
            device for device in self.devices.values()
            if device.device_type.lower() in ["motion_sensor", "motionsensor"]
        ]