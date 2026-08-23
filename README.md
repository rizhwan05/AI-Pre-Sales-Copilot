# Pre-Sales Copilot (RAG) - Exhaustive Documentation

## System Overview
The **Pre-Sales Copilot** is a specialized AI assistant designed to automate and augment the pre-sales workflow. It leverages **Retrieval-Augmented Generation (RAG)** using **LlamaIndex**, **ChromaDB**, and **AWS Bedrock** to analyze Request For Proposals (RFPs) and automatically generate structured Proposals, Estimations, and Statements of Work (SOW) based on historical project data.

---

## 1. Architecture Deep-Dive

### 1.1 Core Components
- **Framework**: FastAPI
- **AI Orchestration**: LlamaIndex Core
- **LLM Engine**: AWS Bedrock (typically Amazon Nova models via `BedrockClient`).
- **Embeddings**: Configured via `embedding_client.py` to vectorize text for similarity search.
- **Vector Database**: Local **ChromaDB** (`./chroma_store`) used to persist historical project case studies, architectures, and proposals.

### 1.2 Data Persistence
1. **Vector Store (`chroma_store/`)**: Holds LlamaIndex semantic chunks. Nodes are inserted with metadata (`project_name`, `tech_stack`, `duration`, `team_size`, `document_type`).
2. **Session Store (`sessions/`)**: When an RFP is uploaded, the parsed requirements are cached in a local JSON file keyed by a UUID `session_id` with a TTL (default 3600 seconds).

---

## 2. API Contract & Workflow

### 2.1 Ingestion Flow
**`POST /ingest/add`**
Used to populate the RAG database with historical documents.
- **Payload (`multipart/form-data`)**:
  - `document_type` (Form string): Must be one of `proposal`, `case_study`, `estimation`, `architecture`.
  - `file` (UploadFile): The document (.txt or .pdf).
- **Behavior**: Extracts text, chunks it semantically, assigns a deterministic SHA256 ID, and persists to ChromaDB.
- **Response**: Returns the number of `indexed_chunks` successfully added.

### 2.2 RFP Upload Flow
**`POST /rfp/upload`**
The entry point for a new pre-sales engagement.
- **Payload (`multipart/form-data`)**:
  - `file` (UploadFile): The raw RFP document.
- **Behavior**: Extracts text and uses the LLM (`extraction_service`) to parse structured requirements (Functional, Non-Functional, Constraints, Compliance, Assumptions). Creates a session.
- **Response**: Returns the `session_id` and the `requirements_summary`.

### 2.3 Follow-Up & Generation Flow
**`POST /rfp/follow-up`**
Used to chat with the copilot or generate specific pre-sales artifacts based on the uploaded RFP.
- **Payload (JSON)**:
  ```json
  {
    "session_id": "<uuid-from-upload>",
    "user_query": "Generate a detailed technical proposal for this RFP"
  }
  ```
- **Behavior**:
  1. Validates the `session_id` and loads the cached requirements.
  2. Detects intent from the `user_query` (e.g., proposal, estimation, SOW, generic).
  3. **If Generic Intent**: Uses RAG. Calls `retrieve_similar_projects` to query ChromaDB, injecting the retrieved historical context alongside the RFP requirements into the LLM prompt.
  4. **If Specific Intent**: Routes to `generate_proposal`, `generate_estimation`, or `generate_statement_of_work` services to create structured markdown documents.
- **Response**:
  ```json
  {
    "output": "<markdown response or generated artifact>"
  }
  ```

---

## 3. Pre-Sales Services
- **Extraction Service**: Translates raw RFP text into categorized dictionaries.
- **Proposal Service**: Generates structured business proposals aligning the RFP needs with the company's capabilities.
- **Estimation Service**: Generates effort, timeline, and cost estimations.
- **SOW Service**: Generates formal Statements of Work.
- **Retrieval Service**: Interfaces with the LlamaIndex `VectorStoreIndex` to find the most relevant past projects based on semantic similarity.

---

## 4. Setup and Execution Steps

### 4.1 Prerequisites
- AWS Bedrock access configured locally.
- Python 3.11+ and `uv`.

### 4.2 Configuration
Set up your environment variables (or `.env` file) for AWS Bedrock access:
```env
AWS_REGION=us-east-1
NOVA_MODEL_ID=amazon.nova-pro-v1:0
```

### 4.3 Start the Server
1. Install dependencies:
   ```bash
   uv sync
   ```
2. *(Optional)* Seed the database with historical data using the CLI script:
   ```bash
   python -m app.ingestion.ingest_projects
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
