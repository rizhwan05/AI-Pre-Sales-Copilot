from __future__ import annotations

import logging
from typing import Dict

from app.services.estimation_service import generate_estimation
from app.services.extraction_service import extract_requirements
from app.services.proposal_service import generate_proposal

logger = logging.getLogger(__name__)


def run_full_presales_pipeline(rfp_text: str) -> Dict[str, object]:
	logger.info("Starting presales pipeline.")
	if not rfp_text or not rfp_text.strip():
		raise ValueError("rfp_text must be a non-empty string")

	requirements = extract_requirements(rfp_text)
	proposal = generate_proposal(requirements)
	effort_estimate = generate_estimation(requirements)

	logger.info("Presales pipeline completed.")
	return {
		"requirements": requirements,
		"proposal": proposal,
		"effort_estimate": effort_estimate,
	}


def handle_followup(structured_requirements: Dict[str, object], user_intent: str) -> Dict[str, object]:
	if not structured_requirements:
		raise ValueError("structured_requirements must be provided")
	if not user_intent or not user_intent.strip():
		raise ValueError("user_intent must be a non-empty string")

	intent = user_intent.strip().lower()
	intent_map = {
		"proposal": "proposal",
		"estimation": "estimation",
		"case study": "case_study",
		"architecture": "architecture",
	}
	if intent not in intent_map:
		raise ValueError("Unsupported user_intent")

	if intent == "proposal":
		return generate_proposal(structured_requirements, document_type=intent_map[intent])
	if intent == "estimation":
		return generate_estimation(structured_requirements)

	return generate_proposal(structured_requirements, document_type=intent_map[intent])
