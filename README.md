# Pre-Sales Copilot (RAG)

## Overview
This project contains a FastAPI backend for an AI pre-sales copilot leveraging Retrieval-Augmented Generation (RAG). It uses ChromaDB for vector storage and AWS Bedrock for Large Language Model (LLM) generation.

## Architecture
The system follows a typical RAG (Retrieval-Augmented Generation) architecture tailored for the Pre-Sales domain.
- **Web Framework**: FastAPI handles HTTP requests.
- **Vector Database**: ChromaDB stores document embeddings (past successful proposals and statements of work) locally in `chroma_store/`.
- **AI/LLM Provider**: AWS Bedrock is accessed via `LlamaIndex` to generate high-quality text for new proposals based on retrieved context.
- **Session Management**: An in-memory temporary cache backed by the filesystem (`sessions/`) stores user intent and extracted requirements for a short TTL (1 hour).

## Application Flow

### 1. Ingestion Flow (Admin/Setup)
1. User uploads a historical project document (like a past Proposal, SOW, or Architecture document).
2. The `file_handler` extracts the text.
3. `ingestion_router` chunks the text, creates embeddings via AWS Bedrock (`embedding_client`), and indexes it into ChromaDB.

### 2. RFP processing Flow (User)
1. **Upload RFP**: The user uploads a Request for Proposal (RFP) document.
2. **Extraction**: The system extracts requirements (Functional, Non-Functional, Constraints, etc.) and returns a `session_id`.
3. **Follow-up Generation**: Using the `session_id` and a user prompt (e.g., "Generate a proposal" or "Generate an estimation"), the orchestrator:
   - Identifies the intent.
   - Retrieves similar past projects from ChromaDB.
   - Passes the extracted requirements + historical context to the LLM (AWS Bedrock).
   - Returns the generated document.

## Endpoints

### Ingestion API
- `POST /api/v1/ingest/add`
  - **Payload**: `multipart/form-data` containing `document_type` and `file`.
  - **Description**: Extracts text from the uploaded file and ingests it into the Chroma vector store for RAG.

### RFP API
- `POST /api/v1/rfp/upload`
  - **Payload**: `multipart/form-data` containing `file`.
  - **Description**: Parses the RFP, extracts structured requirements, saves them to a session, and returns `session_id` and `requirements_summary`.
- `POST /api/v1/rfp/follow-up`
  - **Payload**: JSON `{"session_id": "uuid", "user_query": "string"}`.
  - **Description**: Generates an output (proposal, estimation, SOW, or generic response) using the session requirements and retrieved historical context.

## Usage Steps
1. Ensure your AWS credentials are configured in your environment to allow access to AWS Bedrock.
2. Install dependencies using `uv` (or pip): `uv sync`.
3. Start the FastAPI server:
   ```bash
   python app/main.py
   # Or using uvicorn directly
   uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
   ```
4. Access the API documentation at `http://127.0.0.1:8080/docs`.
5. Use the `/ingest/add` endpoint to populate your ChromaDB with past projects.
6. Use the `/rfp/upload` endpoint with a sample RFP, then use the `/rfp/follow-up` endpoint to generate documents.
