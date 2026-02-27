from __future__ import annotations

import os
from llama_index.vector_stores.chroma import ChromaVectorStore

PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "presales_projects"
def get_vector_store() -> ChromaVectorStore:
	try:
		# FIX 3: Do not initialize embeddings here.
		os.makedirs(PERSIST_DIR, exist_ok=True)
		# FIX 4: Remove misleading metadata flags; Chroma stores metadata by default.
		return ChromaVectorStore.from_params(
			collection_name=COLLECTION_NAME,
			persist_dir=PERSIST_DIR,
		)
	except Exception as exc:
		raise RuntimeError("Failed to initialize Chroma vector store") from exc
