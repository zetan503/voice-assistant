"""
Router for Home Assistant integration endpoints.
Provides API endpoints for interacting with the Home Assistant integration.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.integrations.home_assistant.manager import HomeAssistantManager
from app.integrations.security.perimeter import SecurityEvent

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/hass",
    tags=["home_assistant"],
    responses={404: {"description": "Not found"}},
)

# Store the manager instance
_hass_manager = None


def get_hass_manager() -> HomeAssistantManager:
    """
    Get or create the Home Assistant manager.

    Returns:
        Home Assistant manager instance
    """
    global _hass_manager
    if _hass_manager is None:
        _hass_manager = HomeAssistantManager()
    return _hass_manager


# Request/Response models
class StatusResponse(BaseModel):
    """Response model for integration status."""
    is_initialized: bool
    is_active: bool
    hass_connected: bool
    security_status: Dict[str, Any]


class MessageRequest(BaseModel):
    """Request model for sending announcements."""
    message: str
    priority: bool = False


class ResetZoneRequest(BaseModel):
    """Request model for resetting security zones."""
    zone_id: str


# API endpoints
@router.get("/status", response_model=StatusResponse)
async def get_status(
    manager: HomeAssistantManager = Depends(get_hass_manager)
):
    """Get status of the Home Assistant integration."""
    hass_connected = manager.hass_client is not None and manager.hass_client.is_connected
    security_status = await manager.get_security_status() if manager.is_initialized else {}

    return StatusResponse(
        is_initialized=manager.is_initialized,
        is_active=manager.is_active,
        hass_connected=hass_connected,
        security_status=security_status
    )


@router.post("/initialize")
async def initialize_integration(
    manager: HomeAssistantManager = Depends(get_hass_manager)
):
    """Initialize the Home Assistant integration."""
    if manager.is_initialized:
        return {"message": "Integration already initialized"}

    success = await manager.initialize()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to initialize integration")

    return {"message": "Integration initialized successfully"}


@router.post("/start")
async def start_integration(
    manager: HomeAssistantManager = Depends(get_hass_manager)
):
    """Start the Home Assistant integration."""
    if not manager.is_initialized:
        success = await manager.initialize()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize integration")

    if manager.is_active:
        return {"message": "Integration already active"}

    success = await manager.start()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start integration")

    return {"message": "Integration started successfully"}


@router.post("/stop")
async def stop_integration(
    manager: HomeAssistantManager = Depends(get_hass_manager)
):
    """Stop the Home Assistant integration."""
    if not manager.is_active:
        return {"message": "Integration not active"}

    success = await manager.stop()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop integration")

    return {"message": "Integration stopped successfully"}


@router.post("/announce")
async def make_announcement(
    request: MessageRequest,
    manager: HomeAssistantManager = Depends(get_hass_manager)
):
    """Make an announcement."""
    if not manager.is_initialized:
        raise HTTPException(status_code=400, detail="Integration not initialized")

    success = await manager.make_announcement(request.message, request.priority)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to make announcement")

    return {"message": "Announcement queued successfully"}


@router.post("/reset_zone")
async def reset_security_zone(
    request: ResetZoneRequest,
    manager: HomeAssistantManager = Depends(get_hass_manager)
):
    """Reset a security zone to secure status."""
    if not manager.is_initialized:
        raise HTTPException(status_code=400, detail="Integration not initialized")

    success = await manager.reset_security_zone(request.zone_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Zone {request.zone_id} not found")

    return {"message": f"Zone {request.zone_id} reset successfully"}


@router.get("/sensors")
async def get_sensors(
    manager: HomeAssistantManager = Depends(get_hass_manager)
):
    """Get sensor states from Home Assistant."""
    if not manager.is_initialized:
        raise HTTPException(status_code=400, detail="Integration not initialized")

    if not manager.hass_client.is_connected:
        raise HTTPException(status_code=400, detail="Not connected to Home Assistant")

    sensors = await manager.get_sensor_states()
    return {"sensors": sensors}


@router.get("/security/events")
async def get_security_events(
    limit: int = 10,
    manager: HomeAssistantManager = Depends(get_hass_manager)
):
    """Get recent security events."""
    if not manager.is_initialized:
        raise HTTPException(status_code=400, detail="Integration not initialized")

    events = await manager.perimeter_monitor.get_recent_events(limit)
    return {"events": [event.dict() for event in events]}


@router.get("/camera/status")
async def get_camera_status(
    manager: HomeAssistantManager = Depends(get_hass_manager)
):
    """Get camera status."""
    if not manager.is_initialized:
        raise HTTPException(status_code=400, detail="Integration not initialized")

    status = await manager.get_camera_status()
    return status


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time security events."""
    manager = get_hass_manager()

    await websocket.accept()

    try:
        # Send initial status
        if manager.is_initialized:
            status = await manager.get_security_status()
            await websocket.send_json({"event": "status", "data": status})

        # Placeholder for a more sophisticated event streaming mechanism
        # In a real implementation, we would set up event listeners and forward events to the WebSocket

        while True:
            data = await websocket.receive_json()
            command = data.get("command")

            if command == "get_status":
                status = await manager.get_security_status()
                await websocket.send_json({"event": "status", "data": status})

            elif command == "make_announcement":
                message = data.get("message", "")
                priority = data.get("priority", False)
                if message:
                    await manager.make_announcement(message, priority)
                    await websocket.send_json({"event": "announcement_sent", "message": message})

            await websocket.send_json({"event": "ack", "command": command})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")