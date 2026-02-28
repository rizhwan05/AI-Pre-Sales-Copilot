from fastapi import FastAPI

from app.routers.ingestion_router import ingest_router
from app.routers.rfp_router import rfp_router

app = FastAPI(title="AI Pre-Sales Copilot")

app.include_router(ingest_router, prefix="/api/v1")
app.include_router(rfp_router, prefix="/api/v1")

if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)