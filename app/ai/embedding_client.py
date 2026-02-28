from __future__ import annotations

import logging
from functools import lru_cache

from llama_index.embeddings.bedrock import BedrockEmbedding

BEDROCK_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> BedrockEmbedding:
	# FIX 3: Centralize and reuse the embedding model instance.
	try:
		logger.info("Bedrock client initialized using bearer token authentication")
		return BedrockEmbedding(
			model_id=BEDROCK_EMBED_MODEL_ID,
			region_name="us-east-1",
		)
	except Exception as exc:
		raise RuntimeError("Failed to initialize Bedrock embedding model") from exc
