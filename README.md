# Voice Assistant

A voice assistant application that uses local LLM (Llama/Mixtral) and Whisper models for speech processing. The system provides wake word detection, speech-to-text, text-to-speech capabilities, and can run entirely locally on a machine with GPU support.

## Features

- Local LLM (Llama/Mixtral) running on GPU
- Whisper STT and TTS models for voice processing
- Async processing pipeline with streaming input/output
- Wake word detection mechanism
- Concurrent speech input/output handling
- FastAPI web interface
- Home Assistant integration with real-time security monitoring
- YoLink motion sensor integration for alerts
- Camera integration with facial recognition
- Automated announcements for security events and visitors
- Real-time perimeter security monitoring

## Requirements

- Python 3.8+
- CUDA-capable GPU (for optimal performance)
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/voice-assistant.git
cd voice-assistant
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Configuration settings are stored in `config/settings.py`. You can customize:

- Model paths and parameters
- Audio settings
- API configuration

## Usage

### Web API Mode

Start the web API server:

```bash
python main.py
```

The server will run at http://localhost:8000 by default.

### CLI Mode

Start the voice assistant in CLI mode:

```bash
python main.py --cli
```

In CLI mode, the assistant will listen for wake words and respond to voice commands.

## API Endpoints

- `GET /`: Basic information
- `GET /status`: Get current status
- `POST /start`: Start the voice assistant
- `POST /stop`: Stop the voice assistant
- `POST /transcribe`: Process text input
- `POST /config`: Update configuration
- `WebSocket /ws`: Real-time communication

## WebSocket Events

- `wake_word_detected`: When wake word is detected
- `transcription`: When speech is transcribed
- `response`: When LLM generates a response
- `error`: When an error occurs
- `status`: Current status information

## License

MIT
