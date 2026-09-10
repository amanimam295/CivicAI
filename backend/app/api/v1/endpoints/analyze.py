import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from app.services.document_parser import parse_document
from app.services.cloudinary_service import upload_document
from app.services.snowflake_service import log_analysis_session
from app.services.ai_provider import get_provider
from app.core.state import session_store, document_store
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("")
async def analyze_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form("English"),
    provider: str = Form("gemma-local"),
):
    # 1. Validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")

    # Generate session ID early so both Cloudinary and session storage share the exact same ID
    session_id = str(uuid.uuid4())
    filename = file.filename or "document"

    # Cloudinary upload (fail-soft)
    upload_result = upload_document(content, filename, session_id)

    # 2. Extract text (PyMuPDF with OCR fallback)
    try:
        extracted_text = await parse_document(content, filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to parse document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {str(e)}")

    if not extracted_text.strip():
        raise HTTPException(
            status_code=400, 
            detail="Could not extract any readable text from the document. Please provide a clear PDF or image."
        )

    # 3. Construct prompt
    prompt = f"""You are CivicAI, an Indian government welfare assistant. Your ONLY job is to analyse official Indian government scheme or welfare documents.

SCOPE CHECK — Before doing anything else:
- If the document text below is NOT an official Indian government scheme, welfare notice, entitlement letter, or public-service document (e.g. if it is about election results, news, sports, private contracts, or anything unrelated to government welfare), you MUST immediately return the refusal JSON and STOP.
- NEVER produce output about election results, political outcomes, news events, or any off-topic content.

Document Text:
---
{extracted_text}
---

If the document IS a valid government scheme/welfare document, perform these tasks:
1. Explain what this document is and what it says in plain language (strictly based on the document text above, no external assumptions).
2. Assess the user's eligibility for the scheme or action mentioned in the document.
3. Provide a step-by-step checklist of actions the user needs to take (with deadlines if mentioned in the document).
4. List any missing documents the user still needs to provide.

IMPORTANT: Translate the ENTIRE response into {language}.
IMPORTANT: Base your analysis ONLY on the document text provided above. Do not make up or assume any information not present in the document.
"""

    # 4. Call Provider
    ai_service = get_provider(provider)
    try:
        result = await ai_service.generate_json(prompt)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"JSON generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze document with {provider} AI: {str(e)}",
        )

    # 5. Store session history and uploaded document reference
    session_store[session_id] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(result)},
    ]
    document_store[session_id] = upload_result

    # Snowflake Integration: Log the session fail-softly in background
    background_tasks.add_task(
        log_analysis_session,
        session_id,
        filename,
        language,
        provider,
        result,
    )

    # 6. Return exact expected schema + session_id + document_url
    result["session_id"] = session_id
    result["document_url"] = upload_result.get("url") if upload_result else None

    return result

