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
        Generate a response from the model for a given prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        try:
            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": self.config["max_new_tokens"],
                "temperature": self.config["temperature"],
                "top_p": self.config["top_p"],
                "do_sample": True,
                **kwargs
            }

            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

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

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

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