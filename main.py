"""
Main entry point for the voice assistant application.
Provides a FastAPI interface for interacting with the voice assistant.
"""
import asyncio
import logging
import sys
import os
import signal
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

from app.utils.pipeline import AsyncVoicePipeline
from app.utils.audio import AudioIO
from app.models.llm import LocalLLM
from app.models.speech_to_text import SpeechToText
from app.models.text_to_speech import TextToSpeech
from app.models.wake_word import WakeWordDetector

from config.settings import API, DEBUG

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("voice_assistant.log")
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Voice Assistant API",
    description="API for interacting with the voice assistant",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create voice pipeline
voice_pipeline = AsyncVoicePipeline()

# Define request models
class TextRequest(BaseModel):
    text: str

class AudioRequest(BaseModel):
    audio_base64: str
    sample_rate: int = 16000

class ConfigRequest(BaseModel):
    llm_config: Optional[Dict] = None
    stt_config: Optional[Dict] = None
    tts_config: Optional[Dict] = None
    wakeword_config: Optional[Dict] = None
    audio_config: Optional[Dict] = None

# Websocket connections
connected_clients = set()

# Events and callbacks
async def on_wake_word():
    """Callback for when wake word is detected."""
    logger.info("Wake word callback triggered")
    # Notify connected clients
    for client in connected_clients:
        await client.send_json({"event": "wake_word_detected"})

async def on_transcription(text: str):
    """Callback for when speech is transcribed."""
    logger.info(f"Transcription callback triggered: {text}")
    # Notify connected clients
    for client in connected_clients:
        await client.send_json({"event": "transcription", "text": text})

async def on_response(text: str):
    """Callback for when LLM generates a response."""
    logger.info(f"Response callback triggered: {text}")
    # Notify connected clients
    for client in connected_clients:
        await client.send_json({"event": "response", "text": text})

async def on_error(error: str):
    """Callback for when an error occurs."""
    logger.error(f"Error callback triggered: {error}")
    # Notify connected clients
    for client in connected_clients:
        await client.send_json({"event": "error", "error": error})

# Set up callbacks
voice_pipeline.set_on_wake_word(on_wake_word)
voice_pipeline.set_on_transcription(on_transcription)
voice_pipeline.set_on_response(on_response)
voice_pipeline.set_on_error(on_error)

# Define API routes
@app.get("/")
async def root():
    """Root endpoint providing basic information."""
    return {"message": "Voice Assistant API"}

@app.get("/status")
async def status():
    """Get the current status of the voice assistant."""
    return {
        "is_active": voice_pipeline.is_active,
        "is_listening": voice_pipeline.is_listening,
        "is_speaking": voice_pipeline.is_speaking,
        "is_processing": voice_pipeline.is_processing,
    }

@app.post("/start")
async def start():
    """Start the voice assistant."""
    if not voice_pipeline.is_active:
        # Start in a background task
        asyncio.create_task(voice_pipeline.start())
        return {"message": "Voice assistant started"}
    return {"message": "Voice assistant is already running"}

@app.post("/stop")
async def stop():
    """Stop the voice assistant."""
    if voice_pipeline.is_active:
        await voice_pipeline.stop()
        return {"message": "Voice assistant stopped"}
    return {"message": "Voice assistant is not running"}

@app.post("/transcribe")
async def transcribe(request: TextRequest):
    """Manually trigger the voice assistant with text."""
    if not voice_pipeline.is_processing and not voice_pipeline.is_speaking:
        response = await voice_pipeline.generate_response(request.text)
        asyncio.create_task(voice_pipeline.speak_response(response))
        return {"response": response}
    return JSONResponse(
        status_code=409,
        content={"error": "Voice assistant is busy"}
    )

@app.post("/config")
async def update_config(request: ConfigRequest):
    """Update the configuration of the voice assistant components."""
    # Recreate components with new configurations
    if request.llm_config:
        voice_pipeline.llm = LocalLLM(request.llm_config)

    if request.stt_config:
        voice_pipeline.stt = SpeechToText(request.stt_config)

    if request.tts_config:
        voice_pipeline.tts = TextToSpeech(request.tts_config)

    if request.wakeword_config:
        voice_pipeline.wake_word = WakeWordDetector(request.wakeword_config)

    if request.audio_config:
        voice_pipeline.audio = AudioIO(request.audio_config)

    return {"message": "Configuration updated"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    connected_clients.add(websocket)

    try:
        while True:
            # Receive JSON message
            data = await websocket.receive_json()
            command = data.get("command")

            if command == "start":
                if not voice_pipeline.is_active:
                    asyncio.create_task(voice_pipeline.start())
                    await websocket.send_json({"event": "started"})

            elif command == "stop":
                if voice_pipeline.is_active:
                    await voice_pipeline.stop()
                    await websocket.send_json({"event": "stopped"})

            elif command == "transcribe":
                text = data.get("text")
                if text and not voice_pipeline.is_processing and not voice_pipeline.is_speaking:
                    response = await voice_pipeline.generate_response(text)
                    asyncio.create_task(voice_pipeline.speak_response(response))
                    await websocket.send_json({"event": "response", "text": response})

            elif command == "status":
                await websocket.send_json({
                    "event": "status",
                    "is_active": voice_pipeline.is_active,
                    "is_listening": voice_pipeline.is_listening,
                    "is_speaking": voice_pipeline.is_speaking,
                    "is_processing": voice_pipeline.is_processing,
                })

    except WebSocketDisconnect:
        connected_clients.remove(websocket)

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

# Register shutdown handler
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown handler to clean up resources."""
    if voice_pipeline.is_active:
        await voice_pipeline.stop()
    logger.info("Application shutting down")

# Direct execution entry point
async def run_app():
    """Run the voice assistant application directly (without web API)."""
    # Set up signal handlers for graceful termination
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(voice_pipeline.stop())
        # Give time for cleanup
        asyncio.get_event_loop().call_later(1, asyncio.get_event_loop().stop)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("Starting voice assistant...")
        await voice_pipeline.start()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        await voice_pipeline.stop()

    except Exception as e:
        logger.error(f"Error in voice assistant: {str(e)}")
        await voice_pipeline.stop()

# Main entry point
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Run in CLI mode (direct voice interaction)
        asyncio.run(run_app())
    else:
        # Run as web API
        uvicorn.run(
            "main:app",
            host=API["host"],
            port=API["port"],
            reload=DEBUG
        )