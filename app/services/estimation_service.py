from __future__ import annotations

import json
import logging
from typing import Dict, List

from app.ai.bedrock_client import BedrockClient
from app.ai.prompts import get_estimation_prompt
from app.services.retrieval_service import retrieve_similar_projects

logger = logging.getLogger(__name__)


def generate_estimation(structured_requirements: Dict[str, object]) -> Dict[str, object]:
	if not structured_requirements:
		raise ValueError("structured_requirements must be provided")

	query_text = _summarize_requirements(structured_requirements)
	retrieved = retrieve_similar_projects(query_text, document_type="estimation")
	context = _format_retrieved_context(retrieved)
	context = _truncate_context(context, 6000)

	client = BedrockClient()
	system_prompt, user_prompt = get_estimation_prompt(structured_requirements, context)
	response = client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
	parsed = _safe_parse_json(response)
	if parsed is None:
		raise ValueError("LLM returned invalid JSON")
	return parsed


def _summarize_requirements(requirements: Dict[str, object]) -> str:
	parts: List[str] = []
	for key in (
		"functional_requirements",
		"non_functional_requirements",
		"constraints",
		"compliance_items",
		"assumptions",
	):
		values = requirements.get(key, [])
		if isinstance(values, list) and values:
			parts.append(f"{key}: " + "; ".join(str(value) for value in values))
	return "\n".join(parts).strip()


def _format_retrieved_context(retrieved: List[Dict[str, object]]) -> str:
	if not retrieved:
		return "No estimation references found."

	lines: List[str] = []
	for item in retrieved:
		lines.append(
			" | ".join(
				part
				for part in [
					f"project_name: {item.get('project_name')}",
					f"tech_stack: {item.get('tech_stack')}",
					f"duration: {item.get('duration')}",
					f"team_size: {item.get('team_size')}",
					f"score: {item.get('aggregated_score')}",
				]
				if part
			)
		)
	return "\n".join(lines)


def _safe_parse_json(text: str) -> Dict[str, object] | None:
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		return None


def _truncate_context(context: str, max_chars: int) -> str:
	if len(context) <= max_chars:
		return context
	logger.warning("Truncated retrieved context from %s to %s characters", len(context), max_chars)
	return context[:max_chars].rstrip()
