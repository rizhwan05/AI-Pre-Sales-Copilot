from __future__ import annotations

import logging
from typing import Dict, List

from app.services.estimation_service import estimate_effort
from app.services.extraction_service import extract_requirements
from app.services.proposal_service import generate_proposal
from app.services.retrieval_service import retrieve_similar_projects

logger = logging.getLogger(__name__)


def run_full_presales_pipeline(rfp_text: str) -> Dict[str, object]:
	logger.info("Starting presales pipeline.")
	if not rfp_text or not rfp_text.strip():
		raise ValueError("rfp_text must be a non-empty string")

	requirements = extract_requirements(rfp_text)
	similar_projects = retrieve_similar_projects(rfp_text)
	proposal = generate_proposal(rfp_text, requirements, similar_projects)
	effort_estimate = estimate_effort(rfp_text, requirements, proposal)

	logger.info("Presales pipeline completed.")
	return {
		"requirements": requirements,
		"similar_projects": similar_projects,
		"proposal": proposal,
		"effort_estimate": effort_estimate,
	}
