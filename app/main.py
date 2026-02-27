from __future__ import annotations

import logging

from fastapi import FastAPI

from app.startup_validation import validate_startup

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()


@app.on_event("startup")
def _validate_startup() -> None:
	try:
		validate_startup()
		logger.info("Startup validation succeeded.")
	except Exception as exc:
		logger.exception("Startup validation failed.")
		raise RuntimeError("Startup validation failed.") from exc
