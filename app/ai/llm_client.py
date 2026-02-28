from __future__ import annotations

import logging
import os
from typing import Tuple

from llama_index.llms.bedrock import Bedrock

DEFAULT_LLM_MODEL_ID = "anthropic.claude-opus-4-6"

logger = logging.getLogger(__name__)


def get_llm() -> Bedrock:
	# FIX 6: Central Claude Opus 4.6 initialization for future orchestration.
	try:
		region = _get_region()
		model_id = os.getenv("BEDROCK_LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID)
		llm = Bedrock(
			model=model_id,
			region_name=region,
		)
		logger.info("Bedrock LLM initialized using bearer-token authentication")
		return llm
	except Exception as exc:
		raise RuntimeError("Failed to initialize Bedrock LLM") from exc


def _get_region() -> str:
	region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
	if not region:
		raise ValueError("AWS region must be set via AWS_REGION or AWS_DEFAULT_REGION")
	return region
