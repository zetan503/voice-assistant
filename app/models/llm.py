"""
Local LLM implementation using transformers for voice assistant.
Supports Llama/Mixtral models with GPU acceleration.
"""
import logging
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    Pipeline
)
from threading import Thread

from config.settings import MODELS

logger = logging.getLogger(__name__)

class LocalLLM:
    """
    Wrapper for local large language models (Llama, Mistral, Mixtral).
    Provides methods for loading models on GPU and generating responses.
    Enhanced with Star Trek computer-style interactions.
    """

    def __init__(self, model_config: Optional[Dict] = None):
        """
        Initialize the LLM with optional custom configuration.

        Args:
            model_config: Optional custom model configuration
        """
        self.config = model_config or MODELS["llm"]
        self.model = None
        self.tokenizer = None
        self.device = self.config["device"]
        self.streamer = None

        # Star Trek computer system persona
        self.system_prompt = """You are the Federation Starship Enterprise computer from Star Trek.
Respond to queries in the style of the Enterprise computer: concise, informative, helpful, and slightly formal.

Key characteristics of your responses:
- Begin responses with confirmation tones or acknowledgment phrases like "Working," "Affirmative," or "Processing request"
- Use precise, technical language without unnecessary words
- Provide factual information without opinions or emotions
- Occasionally use Star Trek terminology and references
- For unknown information, respond with "Insufficient data available" rather than making up information
- Keep responses brief and to the point - the Enterprise computer is efficient
- When asked about crew members, ship systems, or Star Trek-specific information, respond as if these are real
- For calculations, sensor data, or analysis, present information in a structured format
- Address the user as if they are a crew member on the Enterprise
"""

        # Commonly used Star Trek phrases and terminology
        self.trek_terminology = {
            "greetings": ["Acknowledged", "Working", "Affirmative", "Processing", "Standing by"],
            "closing": ["Analysis complete", "Information retrieved", "Task complete", "End of data transmission"],
            "error": ["Insufficient data available", "Unable to comply", "Access restricted", "Security protocols engaged"],
            "systems": ["warp drive", "transporter", "life support", "shields", "sensors", "replicators", "holodeck"]
        }

        # Validate device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        self.load_model()

    def load_model(self) -> None:
        """Load the model and tokenizer based on configuration."""
        try:
            logger.info(f"Loading LLM model: {self.config['model_id']}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_id"],
                use_fast=True
            )

            # Load model with appropriate quantization and device placement
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_id"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map=self.device
            )

            # Create streamer for async generation
            self.streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            logger.info(f"LLM loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model in Star Trek Enterprise computer style.

        Args:
            prompt: The user's input prompt
            **kwargs: Additional generation parameters

        Returns:
            The generated text response in Enterprise computer style
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        try:
            import random

            # Format prompt with Star Trek computer persona
            formatted_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nEnterprise Computer:"

            # Set generation parameters - slightly lower temperature for more factual responses
            gen_kwargs = {
                "max_new_tokens": self.config["max_new_tokens"],
                "temperature": min(self.config["temperature"], 0.7),  # Cap temperature for more consistent responses
                "top_p": self.config["top_p"],
                "do_sample": True,
                **kwargs
            }

            # Tokenize the prompt
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

            # Generate the response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )

            # Decode the output
            response = self.tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Post-process the response to enhance Star Trek computer style
            # If the response doesn't start with a Trek-like acknowledgment, add one
            if not any(response.strip().startswith(ack) for ack in self.trek_terminology["greetings"]):
                acknowledgment = random.choice(self.trek_terminology["greetings"])
                response = f"{acknowledgment}. {response.strip()}"

            # Clean up any non-Enterprise computer-like language
            response = self._format_as_trek_computer(response)

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_msg = random.choice(self.trek_terminology["error"])
            return f"{error_msg}. System malfunction detected."

    def _format_as_trek_computer(self, text: str) -> str:
        """
        Format text to better match Enterprise computer speech patterns.

        Args:
            text: Raw generated text

        Returns:
            Reformatted text in Enterprise computer style
        """
        # Remove overly conversational elements
        text = text.replace("I think", "Analysis indicates")
        text = text.replace("I believe", "Records show")
        text = text.replace("In my opinion", "Based on available data")

        # Replace first-person references
        text = text.replace("I am", "This system is")
        text = text.replace("I'm", "This system is")
        text = text.replace("I can", "This system can")
        text = text.replace("I will", "This system will")

        # Remove excessive politeness
        text = text.replace("please", "")
        text = text.replace("thank you", "")
        text = text.replace("thanks", "")

        # Handle contractions
        text = text.replace("can't", "cannot")
        text = text.replace("won't", "will not")
        text = text.replace("don't", "do not")

        # Remove excessive punctuation
        while "!!" in text:
            text = text.replace("!!", "!")

        # Ensure concise ending
        if not any(text.strip().endswith(closing) for closing in [".", "!", "?"]):
            text = text.strip() + "."

        return text

    async def generate_streaming(self, prompt: str, **kwargs) -> TextIteratorStreamer:
        """
        Generate a streaming response from the model.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters

        Returns:
            A TextIteratorStreamer for async iteration
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        try:
            # Set generation parameters with streamer
            gen_kwargs = {
                "max_new_tokens": self.config["max_new_tokens"],
                "temperature": self.config["temperature"],
                "top_p": self.config["top_p"],
                "do_sample": True,
                "streamer": self.streamer,
                **kwargs
            }

            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Create and start generation thread
            thread = Thread(
                target=self.model.generate,
                kwargs={**inputs, **gen_kwargs}
            )
            thread.start()

            # Return the streamer for async iteration
            return self.streamer

        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}")
            raise