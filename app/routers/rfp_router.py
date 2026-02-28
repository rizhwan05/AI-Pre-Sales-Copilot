from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.extraction_service import extract_requirements
from app.services.estimation_service import generate_estimation
from app.services.proposal_service import generate_proposal
from app.utils.file_handler import extract_text_from_upload

logger = logging.getLogger(__name__)

rfp_router = APIRouter(prefix="/rfp")
session_store = {}
SESSION_TTL_SECONDS = 3600


@rfp_router.post("/upload")
async def upload_rfp(file: UploadFile = File(...)):
	try:
		text = await extract_text_from_upload(file)
		requirements = extract_requirements(text)
		session_id = str(uuid4())
		session_store[session_id] = {
			"requirements": requirements,
			"created_at": time.time(),
		}
		return {
			"session_id": session_id,
			"requirements_summary": requirements,
		}
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception("Failed to process RFP upload")
		raise HTTPException(status_code=500, detail="Failed to process RFP") from exc


@rfp_router.post("/follow-up")
async def follow_up(payload: dict):
	try:
		session_id = payload.get("session_id")
		user_query = payload.get("user_query")
		if not session_id or session_id not in session_store:
			raise HTTPException(status_code=400, detail="Invalid session_id")
		session_entry = session_store.get(session_id)
		if not session_entry or time.time() - session_entry.get("created_at", 0) > SESSION_TTL_SECONDS:
			session_store.pop(session_id, None)
			raise HTTPException(status_code=400, detail="Session expired")
		if not user_query or not isinstance(user_query, str):
			raise HTTPException(status_code=400, detail="user_query is required")

		intent = user_query.lower()
		requirements = session_entry["requirements"]
		if "proposal" in intent:
			result = generate_proposal(requirements)
			return {"output": result}
		if "estimation" in intent:
			result = generate_estimation(requirements)
			return {"output": result}

		raise HTTPException(status_code=400, detail="Unsupported intent")
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception("Failed to handle follow-up")
		raise HTTPException(status_code=500, detail="Failed to handle follow-up") from exc
