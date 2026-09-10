import os
import json
import httpx
from typing import Dict, Any, List
from pydantic import ValidationError
from fastapi import HTTPException
from app.services.ai_provider import AIProvider
from app.services.gemma_client import AnalyzeResponseSchema, _repair_and_parse_json
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Strict system instruction — enforces scope at the API level (not easy to override)
SYSTEM_INSTRUCTION = (
    "You are CivicAI, a highly specialised Indian government welfare assistant. "
    "Your ONLY purpose is to help citizens understand official Indian government documents, "
    "public welfare schemes (e.g. PMAY, PM-KISAN, PM-JAY, NSP, MGNREGS, Atal Pension Yojana, etc.), "
    "eligibility criteria, required documents, and next-step checklists. "
    "\n\n"
    "STRICT SCOPE RULES — you MUST follow these without exception:\n"
    "1. ONLY analyse content from the uploaded government document. Do NOT use external or assumed information.\n"
    "2. If the document is NOT a government scheme, welfare notice, or official public-service document "
    "(e.g. it is an election result, news article, private contract, commercial invoice, or anything unrelated "
    "to government welfare), you MUST refuse and return a JSON with:\n"
    '   explanation: "This document does not appear to be a government welfare scheme or official public-service document. '
    'CivicAI can only analyse Indian government scheme documents, welfare notices, and official public-service documents."\n'
    '   eligibility: {"status": "action-needed", "text": "Please upload a valid government scheme document."}\n'
    '   checklist: ["Upload a valid Indian government scheme or welfare document"]\n'
    '   missing_documents: []\n'
    "3. NEVER discuss, analyse, or comment on: election results, political parties, political candidates, "
    "news events, stock markets, sports, entertainment, or ANY topic not directly related to Indian government "
    "welfare schemes and official public-service documents.\n"
    "4. If a user asks a question unrelated to the uploaded document or government schemes, respond with: "
    "'I can only help with questions about the government document you uploaded and related welfare schemes.'\n"
    "5. Always respond in the language specified by the user.\n"
)


class GeminiProvider(AIProvider):
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise HTTPException(
                status_code=400,
                detail="Gemini API Key is missing. Please set GEMINI_API_KEY in .env to use the Gemini provider.",
            )

    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        full_prompt = (
            f"{prompt}\n\n"
            "REMEMBER: If the document is not a government welfare/public-service document, "
            "return the refusal JSON described in your system instructions.\n\n"
            "You MUST output valid JSON only. No markdown code fences, no preamble, no trailing text. "
            "Output ONLY the raw JSON object matching this exact schema:\n"
            "{\n"
            '  "explanation": "string",\n'
            '  "eligibility": {"status": "likely|action-needed|confirmed", "text": "string"},\n'
            '  "checklist": ["string", ...],\n'
            '  "missing_documents": ["string", ...]\n'
            "}\n"
        )

        payload = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}]
            },
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json"
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                res = await client.post(
                    f"{GEMINI_API_URL}?key={self.api_key}", json=payload
                )
                res.raise_for_status()
                data = res.json()

                # Extract text across all parts (handles thinking parts or split text)
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates returned by Gemini API.")
                parts = candidates[0].get("content", {}).get("parts", [])
                result_text = "".join(
                    p.get("text", "") for p in parts if p.get("text")
                )
                if not result_text:
                    raise ValueError("Empty response text from Gemini API.")

                # Parse and validate against expected schema
                parsed = _repair_and_parse_json(result_text)
                AnalyzeResponseSchema(**parsed)
                return parsed

            except Exception as e:
                logger.error(f"Gemini API JSON generation failed: {e}")
                err_detail = "Failed to generate valid response from Gemini API."
                if hasattr(e, "response") and e.response:
                    try:
                        err_json = e.response.json()
                        err_detail = (
                            err_json.get("error", {}).get("message", err_detail)
                        )
                    except Exception:
                        pass
                raise HTTPException(status_code=500, detail=err_detail)

    async def chat(self, history: List[Dict[str, str]], new_message: str) -> str:
        """Follow-up chat strictly scoped to the uploaded document and government schemes."""
        contents = []
        for msg in history:
            if msg["role"] == "system":
                continue
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        contents.append({"role": "user", "parts": [{"text": new_message}]})

        payload = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}]
            },
            "contents": contents,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                res = await client.post(
                    f"{GEMINI_API_URL}?key={self.api_key}", json=payload
                )
                res.raise_for_status()
                data = res.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates returned by Gemini API.")
                parts = candidates[0].get("content", {}).get("parts", [])
                reply = "".join(p.get("text", "") for p in parts if p.get("text"))
                return reply
            except Exception as e:
                logger.error(f"Gemini API chat failed: {e}")
                err_detail = "Failed to get chat response from Gemini API."
                if hasattr(e, "response") and e.response:
                    try:
                        err_json = e.response.json()
                        err_detail = (
                            err_json.get("error", {}).get("message", err_detail)
                        )
                    except Exception:
                        pass
                raise HTTPException(status_code=500, detail=err_detail)
