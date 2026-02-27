from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

from app.ai.embedding_client import get_embedding_model
from app.db.chroma_client import PERSIST_DIR


def validate_startup() -> None:
	_errors = []

	try:
		_get_aws_config()
	except ValueError as exc:
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


def _get_aws_config() -> Tuple[str, str, Optional[str], str]:
	access_key = os.getenv("AWS_ACCESS_KEY_ID")
	secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
	session_token = os.getenv("AWS_SESSION_TOKEN")
	region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

	if not region:
		raise ValueError("AWS region must be set via AWS_REGION or AWS_DEFAULT_REGION")
	if not access_key or not secret_key:
		raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")

	return access_key, secret_key, session_token, region


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
