"""
Configuration settings for the voice assistant application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Models configuration
MODELS = {
    "llm": {
        "model_id": os.getenv("LLM_MODEL_ID", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"),
        "model_basename": os.getenv("LLM_MODEL_BASENAME", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        "device": os.getenv("LLM_DEVICE", "cuda"),
        "max_new_tokens": int(os.getenv("LLM_MAX_NEW_TOKENS", 512)),
        "temperature": float(os.getenv("LLM_TEMPERATURE", 0.7)),
        "top_p": float(os.getenv("LLM_TOP_P", 0.95)),
    },
    "stt": {
        "model_id": os.getenv("STT_MODEL_ID", "openai/whisper-medium"),
        "device": os.getenv("STT_DEVICE", "cuda"),
        "compute_type": os.getenv("STT_COMPUTE_TYPE", "float16"),
    },
    "tts": {
        "model_id": os.getenv("TTS_MODEL_ID", "tts_models/en/ljspeech/tacotron2-DDC"),
        "device": os.getenv("TTS_DEVICE", "cuda"),
    },
    "wakeword": {
        "model_id": os.getenv("WAKEWORD_MODEL_ID", "hey_jarvis"),
        "threshold": float(os.getenv("WAKEWORD_THRESHOLD", 0.5)),
    },
}

# Audio configuration
AUDIO = {
    "sample_rate": int(os.getenv("AUDIO_SAMPLE_RATE", 16000)),
    "channels": int(os.getenv("AUDIO_CHANNELS", 1)),
    "chunk_size": int(os.getenv("AUDIO_CHUNK_SIZE", 1024)),
    "device_index": int(os.getenv("AUDIO_DEVICE_INDEX", 0)),
}

# API configuration
API = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
}

# Debug mode
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")