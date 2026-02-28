from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.utils.file_handler import extract_text_from_upload
from app.ingestion.ingest_projects import ingest_project_text

ingest_router = APIRouter(prefix="/ingest")

@ingest_router.post("/add")
async def ingest(document_type: str = Form(...), file: UploadFile = File(...)):
    try:
        text = extract_text_from_upload(file)
        result = ingest_project_text(project_text=text, metadata={"document_type": document_type})
        return {
            "status": "ingested",
            "document_type": document_type,
            "indexed_chunks": result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc