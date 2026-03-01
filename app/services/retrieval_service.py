"""Retrieve similar projects from Chroma using LlamaIndex."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from app.ai.embedding_client import get_embedding_model
from app.db.chroma_client import get_vector_store

TOP_K = 5

logger = logging.getLogger(__name__)


class VectorStoreEmptyError(RuntimeError):
	"""Raised when the vector store has no indexed documents."""


def retrieve_similar_projects(
	rfp_summary: str,
	document_type: Optional[str] = None,
) -> List[Dict[str, object]]:
	if not rfp_summary or not rfp_summary.strip():
		raise ValueError("Query cannot be empty")

	try:
		# FIX 3: Centralized embedding initialization.
		embed_model = get_embedding_model()

		logger.info(
			"Retrieval request: query_len=%s document_type=%s",
			len(rfp_summary),
			document_type,
		)

		vector_store = get_vector_store()
		if _get_vector_store_count(vector_store) == 0:
			raise VectorStoreEmptyError(
				"Vector store is empty. Ingest documents before querying."
			)
		index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
		retrieval_mode = "semantic"
		if document_type:
			retrieval_mode = "strict"
			logger.info("Strict filtered retrieval for document_type=%s", document_type)
			filters = MetadataFilters(
				filters=[
					ExactMatchFilter(
						key="document_type",
						value=document_type,
					)
				]
			)
			retriever = index.as_retriever(
				similarity_top_k=TOP_K,
				filters=filters,
			)
		else:
			logger.info("Semantic-only retrieval (no document_type filter)")
			retriever = index.as_retriever(similarity_top_k=TOP_K)

		query_embedding = embed_model.get_query_embedding(rfp_summary)
		query_bundle = QueryBundle(query_str=rfp_summary, embedding=query_embedding)
		results = retriever.retrieve(query_bundle)
		logger.info(
			"Retrieval results: mode=%s total_chunks=%s types=%s",
			retrieval_mode,
			len(results),
			_sorted_document_types(results),
		)
		if not results and document_type:
			logger.warning("No results for document_type=%s; retrying without filter", document_type)
			logger.info("Fallback to semantic-only retrieval")
			retriever = index.as_retriever(similarity_top_k=TOP_K)
			results = retriever.retrieve(query_bundle)
			logger.info(
				"Retrieval results: mode=%s total_chunks=%s types=%s",
				"fallback",
				len(results),
				_sorted_document_types(results),
			)
		if not results:
			return []

		return _format_results(results)
	except (VectorStoreEmptyError, ValueError):
		raise
	except Exception as exc:
		raise RuntimeError(f"Failed to retrieve similar projects: {exc}") from exc


def _format_results(results) -> List[Dict[str, object]]:
	grouped: Dict[str, Dict[str, object]] = {}
	for item in results:
		node = getattr(item, "node", None)
		metadata = getattr(node, "metadata", {}) if node else {}
		project_name = metadata.get("project_name") or "Unknown"
		score = getattr(item, "score", None)
		if score is None:
			continue

		current = grouped.get(project_name)
		if current is None:
			current = {
				"project_name": project_name,
				"scores": [],
				"tech_stack": metadata.get("tech_stack"),
				"duration": metadata.get("duration"),
				"team_size": metadata.get("team_size"),
			}
			grouped[project_name] = current

		current["scores"].append(score)
		if not current.get("tech_stack"):
			current["tech_stack"] = metadata.get("tech_stack")
		if not current.get("duration"):
			current["duration"] = metadata.get("duration")
		if not current.get("team_size"):
			current["team_size"] = metadata.get("team_size")

	aggregated: List[Dict[str, object]] = []
	for project in grouped.values():
		scores = sorted(project["scores"], reverse=True)
		max_score = scores[0]
		top3 = scores[:3]
		avg_top3 = sum(top3) / len(top3)
		final_score = max_score + (0.3 * avg_top3)
		aggregated.append(
			{
				"project_name": project["project_name"],
				"aggregated_score": final_score,
				"tech_stack": project.get("tech_stack"),
				"duration": project.get("duration"),
				"team_size": project.get("team_size"),
				"number_of_matching_chunks": len(scores),
			}
		)

	if not aggregated:
		return []

	min_score = min(item["aggregated_score"] for item in aggregated)
	max_score = max(item["aggregated_score"] for item in aggregated)
	denominator = max_score - min_score
	for item in aggregated:
		if denominator == 0:
			item["confidence_score"] = 1.0
		else:
			item["confidence_score"] = (item["aggregated_score"] - min_score) / denominator

	return sorted(
		aggregated,
		key=lambda item: item.get("aggregated_score") or 0,
		reverse=True,
	)[:TOP_K]


def _get_vector_store_count(vector_store) -> int:
	collection = getattr(vector_store, "_collection", None) or getattr(vector_store, "collection", None)
	if collection is not None and hasattr(collection, "count"):
		return int(collection.count())
	if hasattr(vector_store, "count"):
		return int(vector_store.count())
	return 0


def _sorted_document_types(results) -> List[str]:
	types = set()
	for item in results:
		node = getattr(item, "node", None)
		metadata = getattr(node, "metadata", {}) if node else {}
		doc_type = metadata.get("document_type")
		if doc_type:
			types.add(str(doc_type))
	return sorted(types)


def _empty_response() -> Dict[str, object]:
	return {
		"project_name": None,
		"aggregated_score": 0.0,
		"tech_stack": None,
		"duration": None,
		"team_size": None,
		"number_of_matching_chunks": 0,
	}


