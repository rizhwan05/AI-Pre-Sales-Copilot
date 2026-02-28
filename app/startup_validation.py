from __future__ import annotations

import os
import tempfile
from typing import Tuple

from app.ai.embedding_client import get_embedding_model
from app.db.chroma_client import PERSIST_DIR


def validate_startup() -> None:
	_errors = []

	try:
		_get_bedrock_bearer_token()
	except RuntimeError as exc:
		_errors.append(str(exc))

	try:
		_ensure_chroma_writable(PERSIST_DIR)
	except RuntimeError as exc:
		_errors.append(str(exc))

	try:
		_embed_model_ping()
	except RuntimeError as exc:
		_errors.append(str(exc))

	if _errors:
		raise RuntimeError("Startup validation failed: " + "; ".join(_errors))


def _get_bedrock_bearer_token() -> Tuple[str]:
	bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
	if not bearer_token:
		raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK not configured")
	return (bearer_token,)


def _ensure_chroma_writable(persist_dir: str) -> None:
	try:
		os.makedirs(persist_dir, exist_ok=True)
		with tempfile.NamedTemporaryFile(dir=persist_dir, delete=True):
			pass
	except Exception as exc:
		raise RuntimeError(f"Chroma storage path is not writable: {persist_dir}") from exc


def _embed_model_ping() -> None:
	try:
		embed_model = get_embedding_model()
		embed_model.get_text_embedding("startup validation")
	except Exception as exc:
		raise RuntimeError("Bedrock embedding model is not accessible") from exc
