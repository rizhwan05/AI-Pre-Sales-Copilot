from __future__ import annotations

import logging
import os
from functools import lru_cache

import boto3
from llama_index.embeddings.bedrock import BedrockEmbedding

BEDROCK_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> BedrockEmbedding:
	# FIX 3: Centralize and reuse the embedding model instance.
	try:
		_ensure_bedrock_bearer_token()
		region = _get_region()
		client = boto3.client("bedrock-runtime", region_name=region)
		logger.info("Bedrock client initialized using bearer token authentication")
		return BedrockEmbedding(
			model_name=BEDROCK_EMBED_MODEL_ID,
			region_name=region,
			client=client,
		)
	except Exception as exc:
		raise RuntimeError("Failed to initialize Bedrock embedding model") from exc


def _ensure_bedrock_bearer_token() -> None:
	if not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
		raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK not configured")


def _get_region() -> str:
	return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
