"""
Camera integration with facial recognition capabilities.
Provides functionality for capturing images, detecting faces, and identifying known faces.
"""

import logging
import asyncio
import aiohttp
import base64
import os
import time
import io
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Awaitable, Tuple
from datetime import datetime
from pydantic import BaseModel
import face_recognition
from PIL import Image

from config.settings import CAMERA

logger = logging.getLogger(__name__)


class DetectedFace(BaseModel):
    """Model representing a detected face."""
    face_id: str
    confidence: float
    name: Optional[str] = None
    location: Dict[str, int]
    timestamp: datetime


class CameraManager:
    """Manager for camera and facial recognition."""

    def __init__(
        self,
        stream_url: str = None,
        snapshot_url: str = None,
        username: str = None,
        password: str = None,
        known_faces_dir: str = None,
        detection_interval: int = None,
        confidence_threshold: float = None
    ):
        """
        Initialize the camera manager.

        Args:
            stream_url: URL for the camera stream
            snapshot_url: URL for camera snapshots
            username: Camera username
            password: Camera password
            known_faces_dir: Directory with known face images
            detection_interval: Interval between detection attempts in seconds
            confidence_threshold: Minimum confidence for face recognition
        """
        self.stream_url = stream_url or CAMERA.get("stream_url")
        self.snapshot_url = snapshot_url or CAMERA.get("snapshot_url")
        self.username = username or CAMERA.get("username")
        self.password = password or CAMERA.get("password")
        self.known_faces_dir = known_faces_dir or CAMERA.get("known_faces_dir")
        self.detection_interval = detection_interval or CAMERA.get("detection_interval")
        self.confidence_threshold = confidence_threshold or CAMERA.get("confidence_threshold")

        # Auth
        self.auth = None
        if self.username and self.password:
            self.auth = aiohttp.BasicAuth(self.username, self.password)

        # Known faces
        self.known_face_encodings = []
        self.known_face_names = []

        # Detection task
        self._detection_task = None

        # Event callbacks
        self.on_face_detected = None
        self.on_known_face_detected = None
        self.on_unknown_face_detected = None

        # Session
        self.session = None

        # Detection state
        self.last_detected_faces = []
        self._cooldown_until = 0  # Timestamp

    async def initialize(self) -> bool:
        """
        Initialize the camera integration.

        Returns:
            True if initialization is successful, False otherwise
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()

        # Load known faces
        await self._load_known_faces()

        # Create known faces directory if it doesn't exist
        os.makedirs(self.known_faces_dir, exist_ok=True)

        # Start detection task
        if self._detection_task is None or self._detection_task.done():
            self._detection_task = asyncio.create_task(self._face_detection_loop())

        return True

    async def shutdown(self):
        """Shutdown the camera integration."""
        if self._detection_task and not self._detection_task.done():
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
            self._detection_task = None

        if self.session:
            await self.session.close()
            self.session = None

    async def _load_known_faces(self):
        """Load known faces from the known_faces directory."""
        self.known_face_encodings = []
        self.known_face_names = []

        if not os.path.exists(self.known_faces_dir):
            logger.warning(f"Known faces directory {self.known_faces_dir} does not exist")
            return

        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                try:
                    # Extract name from filename (without extension)
                    name = os.path.splitext(filename)[0]

                    # Load image file
                    image_path = os.path.join(self.known_faces_dir, filename)
                    image = face_recognition.load_image_file(image_path)

                    # Get face encodings
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        # Use the first face found in the image
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                        logger.info(f"Loaded known face: {name}")
                    else:
                        logger.warning(f"No face found in {filename}")

                except Exception as e:
                    logger.error(f"Error loading known face {filename}: {str(e)}")

        logger.info(f"Loaded {len(self.known_face_names)} known faces")

    async def get_snapshot(self) -> Optional[bytes]:
        """
        Get a snapshot from the camera.

        Returns:
            Image data as bytes or None if failed
        """
        if not self.snapshot_url:
            logger.error("Snapshot URL not configured")
            return None

        try:
            async with self.session.get(
                self.snapshot_url,
                auth=self.auth,
                timeout=10
            ) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Failed to get snapshot. Status: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting snapshot: {str(e)}")
            return None

    async def detect_faces(self, image_data: bytes) -> List[DetectedFace]:
        """
        Detect and recognize faces in an image.

        Args:
            image_data: Image data as bytes

        Returns:
            List of detected faces
        """
        try:
            # Convert bytes to numpy array
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)

            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image_np)
            face_encodings = face_recognition.face_encodings(image_np, face_locations)

            detected_faces = []

            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = None
                confidence = 0.0

                if self.known_face_encodings:
                    # Get confidence scores (lower distance = higher confidence)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    # Convert distance to confidence (0-1)
                    confidence = 1 - min(face_distances[best_match_index], 1.0)

                    if confidence >= self.confidence_threshold and matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                # Convert face_location to dict
                top, right, bottom, left = face_location
                location = {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                }

                face_id = f"face_{int(time.time())}_{i}"

                detected_face = DetectedFace(
                    face_id=face_id,
                    confidence=confidence,
                    name=name,
                    location=location,
                    timestamp=datetime.now()
                )

                detected_faces.append(detected_face)

            return detected_faces

        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []

    async def _face_detection_loop(self):
        """Run continuous face detection."""
        logger.info("Starting face detection loop")
        try:
            while True:
                # Skip if we're in cooldown period
                if time.time() < self._cooldown_until:
                    await asyncio.sleep(1)
                    continue

                # Get snapshot
                image_data = await self.get_snapshot()
                if image_data:
                    # Detect faces
                    detected_faces = await self.detect_faces(image_data)

                    # Store last detected faces
                    self.last_detected_faces = detected_faces

                    # Handle detected faces
                    if detected_faces:
                        logger.info(f"Detected {len(detected_faces)} faces")

                        # Call face detected callback
                        if self.on_face_detected:
                            await self.on_face_detected(detected_faces, image_data)

                        # Process each face
                        for face in detected_faces:
                            if face.name:
                                logger.info(f"Recognized face: {face.name} (confidence: {face.confidence:.2f})")
                                if self.on_known_face_detected:
                                    await self.on_known_face_detected(face, image_data)
                            else:
                                logger.info(f"Unknown face detected (confidence: {face.confidence:.2f})")
                                if self.on_unknown_face_detected:
                                    await self.on_unknown_face_detected(face, image_data)

                # Wait for next detection interval
                await asyncio.sleep(self.detection_interval)

        except asyncio.CancelledError:
            logger.info("Face detection loop cancelled")
        except Exception as e:
            logger.error(f"Error in face detection loop: {str(e)}")

    def set_face_detected_callback(self, callback: Callable[[List[DetectedFace], bytes], Awaitable[None]]):
        """
        Set callback for face detection events.

        Args:
            callback: Async function to call when faces are detected
        """
        self.on_face_detected = callback

    def set_known_face_callback(self, callback: Callable[[DetectedFace, bytes], Awaitable[None]]):
        """
        Set callback for known face detection events.

        Args:
            callback: Async function to call when a known face is detected
        """
        self.on_known_face_detected = callback

    def set_unknown_face_callback(self, callback: Callable[[DetectedFace, bytes], Awaitable[None]]):
        """
        Set callback for unknown face detection events.

        Args:
            callback: Async function to call when an unknown face is detected
        """
        self.on_unknown_face_detected = callback

    def set_detection_cooldown(self, seconds: int):
        """
        Set a cooldown period for face detection.

        Args:
            seconds: Cooldown period in seconds
        """
        self._cooldown_until = time.time() + seconds

    async def add_known_face(self, name: str, image_data: bytes) -> bool:
        """
        Add a new known face.

        Args:
            name: Name for the face
            image_data: Image data containing the face

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert bytes to numpy array
            image = Image.open(io.BytesIO(image_data))

            # Save the image
            file_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
            image.save(file_path, "JPEG")

            # Reload known faces
            await self._load_known_faces()

            return True

        except Exception as e:
            logger.error(f"Error adding known face: {str(e)}")
            return False