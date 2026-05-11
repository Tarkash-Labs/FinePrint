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
        response_mime_type: str | None = None,
    ) -> str:
        if not self.api_key:
            raise GemmaClientError("Missing GEMMA_API_KEY")

        if self.client is None:
            self.client = genai.Client(api_key=self.api_key)

        # THE FIX: Gemma models reject the 'system_instruction' config parameter.
        # We manually intercept it and prepend it to the user's prompt.
        final_contents = contents
        if system_instruction:
            if isinstance(contents, str):
                final_contents = f"{system_instruction}\n\n{contents}"
            elif isinstance(contents, list):
                final_contents = [f"{system_instruction}\n\n"] + contents
            else:
                final_contents = f"{system_instruction}\n\n{str(contents)}"

        config_args = {
            "temperature": temperature,
        }
        
        # Enforce JSON output if requested
        if response_mime_type:
            config_args["response_mime_type"] = response_mime_type
            
        config = types.GenerateContentConfig(**config_args)

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=final_contents,
                config=config,
            )
            if not response.text:
                raise GemmaClientError("No text returned by Gemma API")
            return response.text
        except Exception as exc:
            raise GemmaClientError(f"Gemma API error: {exc}")