"""Semantic chunking utility using LlamaIndex."""

from __future__ import annotations

import inspect
import math
from statistics import median
from typing import Dict, List, Optional, Sequence

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode as Node
from llama_index.core.utils import get_tokenizer
from app.ai.embedding_client import get_embedding_model


def semantic_chunk_documents(documents: List[Document]) -> List[Node]:
	if not documents:
		return []

	try:
		embed_model = get_embedding_model()
		# FIX 1: Use dynamic chunk_size; keep buffer_size stable.
		chunk_size = _compute_chunk_size(documents)
		buffer_size = 2
		normalized_docs = _normalize_documents(documents)

		splitter = _build_splitter(embed_model, chunk_size, buffer_size)
		return splitter.get_nodes_from_documents(normalized_docs)
	except Exception as exc:
		raise RuntimeError("Failed to chunk documents with semantic splitter") from exc


def _build_splitter(
	embed_model,
	chunk_size: int,
	buffer_size: int,
) -> SemanticSplitterNodeParser:
	params = inspect.signature(SemanticSplitterNodeParser.__init__).parameters
	kwargs: Dict[str, object] = {
		"embed_model": embed_model,
		"buffer_size": buffer_size,
		"include_metadata": True,
	}
	if "chunk_size" in params:
		kwargs["chunk_size"] = chunk_size
	elif "breakpoint_percentile_threshold" in params:
		# FIX 1: Use breakpoint threshold if chunk_size is unavailable.
		kwargs["breakpoint_percentile_threshold"] = _compute_breakpoint_threshold(chunk_size)
	return SemanticSplitterNodeParser(**kwargs)


def _normalize_documents(documents: Sequence[Document]) -> List[Document]:
	filtered: List[Document] = []
	for doc in documents:
		text = doc.text
		if not text or not text.strip():
			continue
		# FIX 5: Preserve full metadata to retain provenance.
		metadata = _filter_metadata(getattr(doc, "metadata", None))
		filtered.append(Document(text=text, metadata=metadata, id_=getattr(doc, "id_", None)))
	return filtered


def _filter_metadata(metadata: Optional[Dict[str, object]]) -> Dict[str, object]:
	if not metadata:
		return {}
	return dict(metadata)


def _compute_chunk_size(documents: Sequence[Document]) -> int:
	# FIX 1: Token-based chunk sizing using LlamaIndex tokenizer.
	MAX_NODES_PER_DOC = 40
	token_counts = [_token_count(doc.text) for doc in documents if doc.text]
	if not token_counts:
		return 512
	median_tokens = median(token_counts)
	chunk_size = max(256, min(1024, round(median_tokens / 2)))
	max_tokens = max(token_counts)
	min_chunk_for_max_nodes = max(1, math.ceil(max_tokens / MAX_NODES_PER_DOC))
	return max(chunk_size, min_chunk_for_max_nodes)


def _compute_breakpoint_threshold(chunk_size: int) -> int:
	if chunk_size >= 900:
		return 97
	if chunk_size >= 700:
		return 95
	if chunk_size >= 500:
		return 92
	return 90


def _token_count(text: str) -> int:
	if not text:
		return 0
	tokenizer = get_tokenizer()
	return len(tokenizer(text))


