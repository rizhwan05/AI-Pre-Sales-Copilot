"""Ingest past project documents into Chroma using LlamaIndex."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, List, Optional, Sequence

from llama_index.core import Document, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex

from app.ai.embedding_client import get_embedding_model
from app.db.chroma_client import get_vector_store
from app.utils.chunking import semantic_chunk_documents

PERSIST_DIR = "./chroma_store"
DATA_DIR = "./data/past_projects"
_METADATA_KEYS = ("project_name", "tech_stack", "duration", "team_size", "document_type")
ALLOWED_DOCUMENT_TYPES = {"proposal", "case_study", "estimation", "architecture"}

logger = logging.getLogger(__name__)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
	index_documents(DATA_DIR)


def index_documents(data_dir: str) -> None:
	documents = load_documents(data_dir)
	if not documents:
		logger.info("No documents found to ingest.")
		return

	# FIX 3: Centralized embedding initialization.
	embed_model = get_embedding_model()
	Settings.embed_model = embed_model

	vector_store = get_vector_store()
	storage_context = StorageContext.from_defaults(vector_store=vector_store)
	pre_count = _get_vector_store_count(vector_store)

	_remove_existing_docs(vector_store, documents)
	# Ensure metadata is JSON-safe before chunking/indexing.
	documents = [_sanitize_document_metadata(doc) for doc in documents]
	semantic_nodes = semantic_chunk_documents(documents)

	index = VectorStoreIndex.from_vector_store(
		vector_store=vector_store,
		storage_context=storage_context,
	)
	index.insert_nodes(semantic_nodes)
	storage_context.persist(persist_dir=PERSIST_DIR)
	post_count = _get_vector_store_count(vector_store)
	if post_count <= pre_count:
		raise RuntimeError("Vector store count did not increase after ingestion")

	logger.info("Indexed %s documents.", len(documents))


def ingest_project_text(project_text: str, metadata: Dict[str, object]) -> int:
	if not project_text or not project_text.strip():
		logger.info("No project text provided for ingestion.")
		return 0

	# FIX 8: Router-friendly ingestion without file system dependency.
	metadata = dict(metadata or {})
	document_type = metadata.get("document_type")
	if not isinstance(document_type, str) or not document_type.strip():
		raise ValueError("Invalid document_type")
	document_type = document_type.strip().lower()
	if document_type not in ALLOWED_DOCUMENT_TYPES:
		raise ValueError("Invalid document_type")
	metadata.setdefault("project_name", "Unknown")
	metadata.setdefault("tech_stack", "Unknown")
	metadata.setdefault("duration", "Unknown")
	metadata.setdefault("team_size", "Unknown")
	metadata["document_type"] = document_type

	doc_id = _build_text_document_id(project_text, metadata)
	document = Document(text=project_text, metadata=metadata, id_=doc_id)
	document = _sanitize_document_metadata(document)

	# Reuse shared embedding + vector store configuration.
	embed_model = get_embedding_model()
	Settings.embed_model = embed_model

	vector_store = get_vector_store()
	storage_context = StorageContext.from_defaults(vector_store=vector_store)

	_remove_existing_docs(vector_store, [document])
	semantic_nodes = semantic_chunk_documents([document])
	if not semantic_nodes:
		logger.info("No chunks produced for project text ingestion.")
		return 0

	index = VectorStoreIndex.from_vector_store(
		vector_store=vector_store,
		storage_context=storage_context,
	)
	index.insert_nodes(semantic_nodes)
	storage_context.persist(persist_dir=PERSIST_DIR)

	logger.info("Indexed %s chunks from project text.", len(semantic_nodes))
	return len(semantic_nodes)


def load_documents(data_dir: str) -> List[Document]:
	reader = SimpleDirectoryReader(
		input_dir=data_dir,
		required_exts=[".pdf", ".txt"],
		file_metadata=_build_metadata_from_path,
	)
	documents = reader.load_data()
	return _assign_document_ids(documents, data_dir)


def _build_metadata_from_path(file_path: str) -> Dict[str, object]:
	# FIX 2: Preserve file_path and file_name for deterministic IDs.
	metadata = _extract_metadata_from_filename(file_path)
	document_type = metadata.get("document_type")
	if not isinstance(document_type, str) or not document_type.strip():
		raise ValueError("Invalid document_type")
	document_type = document_type.strip().lower()
	if document_type not in ALLOWED_DOCUMENT_TYPES:
		raise ValueError("Invalid document_type")
	metadata["document_type"] = document_type
	metadata["file_path"] = file_path
	metadata["file_name"] = os.path.basename(file_path)
	return {key: value for key, value in metadata.items() if value}


def _extract_metadata_from_filename(file_path: str) -> Dict[str, Optional[str]]:
	base_name = os.path.splitext(os.path.basename(file_path))[0]
	parts = [part for part in base_name.split("_") if part]
	values: Dict[str, Optional[str]] = {key: None for key in _METADATA_KEYS}
	for key, value in zip(_METADATA_KEYS, parts):
		values[key] = value
	return values


def _assign_document_ids(documents: Sequence[Document], data_dir: str) -> List[Document]:
	assigned: List[Document] = []
	for doc in documents:
		# FIX 2: Use file_path-derived IDs to avoid duplicate vectors.
		path = doc.metadata.get("file_path") if doc.metadata else None
		if not path:
			raise ValueError("file_path metadata is required for deterministic document IDs")
		rel_path = os.path.relpath(path, data_dir)
		doc_id = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()
		assigned.append(Document(text=doc.text, metadata=doc.metadata, id_=doc_id))
	return assigned


def _build_text_document_id(project_text: str, metadata: Dict[str, object]) -> str:
	project_name = metadata.get("project_name")
	tech_stack = metadata.get("tech_stack")
	duration = metadata.get("duration")
	if all(isinstance(value, str) and value.strip() for value in (project_name, tech_stack, duration)):
		seed = f"{project_name.strip()}|{tech_stack.strip()}|{duration.strip()}"
		return hashlib.sha256(seed.encode("utf-8")).hexdigest()
	return hashlib.sha256(project_text.strip().encode("utf-8")).hexdigest()


def _get_vector_store_count(vector_store) -> int:
	collection = getattr(vector_store, "_collection", None) or getattr(vector_store, "collection", None)
	if collection is not None and hasattr(collection, "count"):
		return int(collection.count())
	if hasattr(vector_store, "count"):
		return int(vector_store.count())
	return 0


def _sanitize_document_metadata(document: Document) -> Document:
	metadata = dict(getattr(document, "metadata", {}) or {})
	for key in _METADATA_KEYS:
		if key != "document_type":
			metadata.setdefault(key, "Unknown")

	sanitized: Dict[str, object] = {}
	for key, value in metadata.items():
		if value is None:
			continue
		if isinstance(value, (str, int, float, bool)):
			sanitized[key] = str(value) if not isinstance(value, str) else value
			continue
		if isinstance(value, (list, tuple)):
			items = [item for item in value if isinstance(item, (str, int, float, bool))]
			if items:
				sanitized[key] = ", ".join(str(item) for item in items)
			continue
		if isinstance(value, dict):
			pairs = []
			for k, v in value.items():
				if isinstance(k, (str, int, float, bool)) and isinstance(v, (str, int, float, bool)):
					pairs.append(f"{k}:{v}")
			if pairs:
				sanitized[key] = ", ".join(pairs)
			continue

	if not sanitized:
		sanitized = {key: "Unknown" for key in _METADATA_KEYS}

	return Document(text=document.text, metadata=sanitized, id_=getattr(document, "id_", None))


def _remove_existing_docs(vector_store, documents: Sequence[Document]) -> None:
	if not hasattr(vector_store, "delete"):
		return

	for doc in documents:
		doc_id = getattr(doc, "id_", None)
		if not doc_id:
			continue
		try:
			vector_store.delete(ref_doc_id=doc_id)
		except TypeError:
			vector_store.delete(doc_id)
		except Exception:
			logger.warning("Failed to delete existing vectors for %s", doc_id)


if __name__ == "__main__":
	# FIX 7: Supports execution via `python -m app.ingestion.ingest_projects`.
	main()
