from __future__ import annotations

import json
import logging
import time
from pathlib import Path
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
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)


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
		_session_save(session_id, session_store[session_id])
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
		if not session_id:
			raise HTTPException(status_code=400, detail="Invalid session_id")
		session_entry = session_store.get(session_id)
		if session_entry is None:
			session_entry = _session_load(session_id)
			if session_entry is not None:
				session_store[session_id] = session_entry
		if session_entry is None:
			raise HTTPException(status_code=400, detail="Invalid session_id")
		if time.time() - session_entry.get("created_at", 0) > SESSION_TTL_SECONDS:
			session_store.pop(session_id, None)
			_session_delete(session_id)
			raise HTTPException(status_code=400, detail="Session expired")
		if not user_query or not isinstance(user_query, str):
			raise HTTPException(status_code=400, detail="user_query is required")

		intent = user_query.strip().lower()
		requirements = session_entry["requirements"]
		document_type = _detect_document_type(intent)

		if "estimation" in intent:
			result = generate_estimation(requirements, document_type=document_type)
			return {"output": result}

		result = generate_proposal(requirements, document_type=document_type)
		return {"output": result}
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception("Failed to handle follow-up")
		raise HTTPException(status_code=500, detail="Failed to handle follow-up") from exc


def _session_path(session_id: str) -> Path:
	return SESSIONS_DIR / f"{session_id}.json"


def _session_save(session_id: str, session_entry: dict) -> None:
	try:
		path = _session_path(session_id)
		path.write_text(json.dumps(session_entry), encoding="utf-8")
	except Exception:
		logger.warning("Failed to persist session %s", session_id)


def _session_load(session_id: str) -> dict | None:
	path = _session_path(session_id)
	if not path.exists():
		return None
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		logger.warning("Failed to load session %s", session_id)
		return None


def _session_delete(session_id: str) -> None:
	path = _session_path(session_id)
	try:
		if path.exists():
			path.unlink()
	except Exception:
		logger.warning("Failed to delete session %s", session_id)


def _detect_document_type(intent: str) -> str | None:
	intent = intent.strip().lower()
	if "proposal" in intent or "proposals" in intent:
		return "proposal"
	if "estimation" in intent or "estimations" in intent or "estimate" in intent:
		return "estimation"
	if "case study" in intent or "case studies" in intent:
		return "case_study"
	if "architecture" in intent or "architectures" in intent:
		return "architecture"
	return None
