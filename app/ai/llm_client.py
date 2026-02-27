from __future__ import annotations

import os
from typing import Optional, Tuple

from llama_index.llms.bedrock import Bedrock

DEFAULT_LLM_MODEL_ID = "anthropic.claude-opus-4-6"


def get_llm() -> Bedrock:
	# FIX 6: Central Claude Opus 4.6 initialization for future orchestration.
	try:
		access_key, secret_key, session_token, region = _get_aws_config()
		model_id = os.getenv("BEDROCK_LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID)
		return Bedrock(
			model=model_id,
			region_name=region,
			aws_access_key_id=access_key,
			aws_secret_access_key=secret_key,
			aws_session_token=session_token,
		)
	except Exception as exc:
		raise RuntimeError("Failed to initialize Bedrock LLM") from exc


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
