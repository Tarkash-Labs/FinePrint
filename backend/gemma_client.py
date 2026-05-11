from __future__ import annotations

from typing import Any
from google import genai
from google.genai import types

class GemmaClientError(RuntimeError):
    pass

class GemmaClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client: genai.Client | None = None
        if api_key:
            self.client = genai.Client(api_key=api_key)

    def generate_content(
        self,
        model: str,
        contents: list | str | Any,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        if not self.api_key:
            raise GemmaClientError("Missing GEMMA_API_KEY")

        if self.client is None:
            self.client = genai.Client(api_key=self.api_key)

        config_args = {
            "temperature": temperature,
        }
        
        if system_instruction:
            config_args["system_instruction"] = system_instruction
            
        config = types.GenerateContentConfig(**config_args)

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            if not response.text:
                raise GemmaClientError("No text returned by Gemma API")
            return response.text
        except Exception as exc:
            raise GemmaClientError(f"Gemma API error: {exc}")
