import os
import json
import asyncio
from typing import Dict, Any, List
from pydantic import ValidationError
from fastapi import HTTPException
from google import genai
from google.genai import errors as genai_errors
from app.services.ai_provider import AIProvider
from app.services.gemma_client import AnalyzeResponseSchema, _repair_and_parse_json
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# We use the official google-genai SDK rather than raw REST calls. Newer
# "AQ."-prefixed Auth keys (the default AI Studio issues since mid-2026)
# are widely reported to fail raw REST calls to generativelanguage.googleapis.com
# with 401 ACCESS_TOKEN_TYPE_UNSUPPORTED, regardless of whether the key is sent
# via ?key= or the x-goog-api-key header. The official SDK handles the auth
# negotiation correctly, which raw HTTP requests do not currently replicate.
GEMINI_MODEL = "gemini-3.6-flash"

# Retry only on errors worth retrying: overload/rate-limit (429), and
# transient server-side failures (5xx). Auth/bad-request errors (4xx other
# than 429) are not retried since retrying won't fix them.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_DELAY_SECONDS = 2.0

async def _create_interaction_with_retry(client: genai.Client, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await client.aio.interactions.create(**kwargs)
        except genai_errors.APIError as e:
            last_exc = e
            if e.code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                delay = _BASE_DELAY_SECONDS * (2 ** attempt)
                logger.warning(
                    f"Gemini API returned {e.code}, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
                continue
            break
        except Exception as e:
            # Network-level errors (timeouts, connection issues) - retry those too.
            last_exc = e
            if attempt < _MAX_RETRIES:
                delay = _BASE_DELAY_SECONDS * (2 ** attempt)
                logger.warning(
                    f"Gemini API network error, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES}): {e}"
                )
                await asyncio.sleep(delay)
                continue
            break
    raise last_exc

class GeminiProvider(AIProvider):
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise HTTPException(
                status_code=400, 
                detail="Gemini API Key is missing. Please set GEMINI_API_KEY in .env to use the Gemini provider."
            )
        self._client = genai.Client(api_key=self.api_key)

    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        full_prompt = (
            f"{prompt}\n\n"
            "You MUST output valid JSON. No markdown formatting block (like ```json), no preamble, no trailing text. "
            "Just the raw JSON object matching this schema:\n"
            "{\n"
            '  "explanation": "string",\n'
            '  "eligibility": {"status": "likely|action-needed|confirmed", "text": "string"},\n'
            '  "checklist": ["string", "string"],\n'
            '  "missing_documents": ["string", "string"]\n'
            "}\n"
        )

        try:
            interaction = await _create_interaction_with_retry(
                self._client,
                model=GEMINI_MODEL,
                input=full_prompt,
                store=False,  # don't let Google retain uploaded-document content server-side
            )
            result_text = interaction.output_text
            if not result_text:
                raise ValueError("Empty response text from Gemini API.")

            parsed = _repair_and_parse_json(result_text)
            AnalyzeResponseSchema(**parsed)
            return parsed
        except Exception as e:
            logger.error(f"Gemini API JSON generation failed: {e}")
            err_detail = "Failed to generate valid response from Gemini API."
            if isinstance(e, genai_errors.APIError) and e.message:
                err_detail = e.message
            raise HTTPException(status_code=500, detail=err_detail)

    async def chat(self, history: List[Dict[str, str]], new_message: str) -> str:
        # Convert prior turns into Interactions API steps.
        input_steps = []
        for msg in history:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "user":
                input_steps.append({"type": "user_input", "content": msg["content"]})
            else:  # assistant/model
                input_steps.append({
                    "type": "model_output",
                    "content": [{"type": "text", "text": msg["content"]}]
                })

        input_steps.append({"type": "user_input", "content": new_message})

        try:
            interaction = await _create_interaction_with_retry(
                self._client,
                model=GEMINI_MODEL,
                input=input_steps,
                store=False,
            )
            return interaction.output_text or ""
        except Exception as e:
            logger.error(f"Gemini API chat failed: {e}")
            err_detail = "Failed to get chat response from Gemini API."
            if isinstance(e, genai_errors.APIError) and e.message:
                err_detail = e.message
            raise HTTPException(status_code=500, detail=err_detail)
