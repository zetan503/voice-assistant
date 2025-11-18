"""
Home Assistant API client for interacting with Home Assistant instance.
Provides functionality for connecting to Home Assistant, fetching states,
and controlling devices.
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel

from config.settings import HOME_ASSISTANT

logger = logging.getLogger(__name__)


class EntityState(BaseModel):
    """Model representing a Home Assistant entity state."""
    entity_id: str
    state: str
    attributes: Dict[str, Any]
    last_changed: str
    last_updated: str


class HomeAssistantClient:
    """Client for interacting with Home Assistant API."""

    def __init__(self, url: str = None, token: str = None, verify_ssl: bool = None):
        """
        Initialize the Home Assistant client.

        Args:
            url: Home Assistant URL
            token: Long-lived access token
            verify_ssl: Whether to verify SSL certificate
        """
        self.url = url or HOME_ASSISTANT.get("url")
        self.token = token or HOME_ASSISTANT.get("token")
        self.verify_ssl = verify_ssl if verify_ssl is not None else HOME_ASSISTANT.get("verify_ssl")

        if not self.url or not self.token:
            logger.warning("Home Assistant URL or token not provided. Integration will not work.")

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.session = None
        self.is_connected = False
        self.states_cache = {}
        self._last_update = 0

    async def connect(self) -> bool:
        """
        Establish connection with Home Assistant.

        Returns:
            True if connection is successful, False otherwise
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()

        try:
            # Test the connection by fetching the API status
            async with self.session.get(
                f"{self.url}/api/",
                headers=self.headers,
                ssl=self.verify_ssl
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Connected to Home Assistant. Version: {data.get('version')}")
                    self.is_connected = True
                    return True
                else:
                    logger.error(f"Failed to connect to Home Assistant. Status: {response.status}")
                    self.is_connected = False
                    return False
        except Exception as e:
            logger.error(f"Error connecting to Home Assistant: {str(e)}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Close the session and disconnect from Home Assistant."""
        if self.session:
            await self.session.close()
            self.session = None
            self.is_connected = False
            logger.info("Disconnected from Home Assistant")

    async def get_states(self, force_refresh: bool = False) -> List[EntityState]:
        """
        Fetch all entity states from Home Assistant.

        Args:
            force_refresh: Whether to force a refresh or use cached data

        Returns:
            List of entity states
        """
        if not self.is_connected and not await self.connect():
            logger.error("Cannot get states: Not connected to Home Assistant")
            return []

        current_time = asyncio.get_event_loop().time()
        # Refresh cache if it's older than 5 seconds or force_refresh is True
        if force_refresh or (current_time - self._last_update > 5):
            try:
                async with self.session.get(
                    f"{self.url}/api/states",
                    headers=self.headers,
                    ssl=self.verify_ssl
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.states_cache = {item["entity_id"]: item for item in data}
                        self._last_update = current_time
                    else:
                        logger.error(f"Failed to fetch states. Status: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"Error fetching states: {str(e)}")
                return []

        # Convert dict to list of EntityState objects
        return [EntityState(**state) for state in self.states_cache.values()]

    async def get_state(self, entity_id: str) -> Optional[EntityState]:
        """
        Get state of a specific entity.

        Args:
            entity_id: Entity ID to fetch

        Returns:
            EntityState object if found, None otherwise
        """
        if not self.is_connected and not await self.connect():
            logger.error(f"Cannot get state for {entity_id}: Not connected to Home Assistant")
            return None

        try:
            async with self.session.get(
                f"{self.url}/api/states/{entity_id}",
                headers=self.headers,
                ssl=self.verify_ssl
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return EntityState(**data)
                else:
                    logger.error(f"Failed to fetch state for {entity_id}. Status: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching state for {entity_id}: {str(e)}")
            return None

    async def call_service(
        self, domain: str, service: str,
        service_data: Optional[Dict[str, Any]] = None,
        target: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> bool:
        """
        Call a service in Home Assistant.

        Args:
            domain: Service domain
            service: Service name
            service_data: Service data
            target: Target entities

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected and not await self.connect():
            logger.error(f"Cannot call service {domain}.{service}: Not connected to Home Assistant")
            return False

        payload = {}
        if service_data:
            payload["data"] = service_data
        if target:
            payload["target"] = target

        try:
            async with self.session.post(
                f"{self.url}/api/services/{domain}/{service}",
                headers=self.headers,
                json=payload,
                ssl=self.verify_ssl
            ) as response:
                if response.status in [200, 201]:
                    logger.info(f"Service {domain}.{service} called successfully")
                    return True
                else:
                    logger.error(f"Failed to call service {domain}.{service}. Status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error calling service {domain}.{service}: {str(e)}")
            return False

    async def turn_on(self, entity_id: str) -> bool:
        """
        Turn on an entity.

        Args:
            entity_id: Entity ID to turn on

        Returns:
            True if successful, False otherwise
        """
        domain = entity_id.split('.')[0]
        return await self.call_service(domain, 'turn_on', target={"entity_id": entity_id})

    async def turn_off(self, entity_id: str) -> bool:
        """
        Turn off an entity.

        Args:
            entity_id: Entity ID to turn off

        Returns:
            True if successful, False otherwise
        """
        domain = entity_id.split('.')[0]
        return await self.call_service(domain, 'turn_off', target={"entity_id": entity_id})

    async def get_history(
        self, entity_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get history for an entity.

        Args:
            entity_id: Entity ID to fetch history for
            start_time: ISO formatted start time
            end_time: ISO formatted end time

        Returns:
            List of historical state changes
        """
        if not self.is_connected and not await self.connect():
            logger.error(f"Cannot get history for {entity_id}: Not connected to Home Assistant")
            return []

        params = {"filter_entity_id": entity_id}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        try:
            async with self.session.get(
                f"{self.url}/api/history/period",
                headers=self.headers,
                params=params,
                ssl=self.verify_ssl
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch history for {entity_id}. Status: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching history for {entity_id}: {str(e)}")
            return []

    async def fire_event(self, event_type: str, event_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Fire an event in Home Assistant.

        Args:
            event_type: Event type
            event_data: Event data

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected and not await self.connect():
            logger.error(f"Cannot fire event {event_type}: Not connected to Home Assistant")
            return False

        try:
            async with self.session.post(
                f"{self.url}/api/events/{event_type}",
                headers=self.headers,
                json=event_data or {},
                ssl=self.verify_ssl
            ) as response:
                if response.status == 200:
                    logger.info(f"Event {event_type} fired successfully")
                    return True
                else:
                    logger.error(f"Failed to fire event {event_type}. Status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error firing event {event_type}: {str(e)}")
            return False