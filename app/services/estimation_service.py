from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from app.ai.bedrock_client import BedrockClient
from app.ai.prompts import get_estimation_prompt
from app.services.retrieval_service import retrieve_similar_projects

logger = logging.getLogger(__name__)


def generate_estimation(
	structured_requirements: Dict[str, object],
	document_type: Optional[str] = None,
) -> Dict[str, object]:
	if not structured_requirements:
		raise ValueError("structured_requirements must be provided")

	query_text = _summarize_requirements(structured_requirements)
	retrieved = retrieve_similar_projects(query_text, document_type=document_type)
	context = _format_retrieved_context(retrieved)
	context = _truncate_context(context, 6000)
	logger.info(
		"Estimation LLM context: chunks=%s chars=%s empty=%s",
		len(retrieved),
		len(context),
		not context.strip(),
	)
	if not context.strip():
		logger.warning("LLM answering from RFP structured summary only")

	client = BedrockClient()
	system_prompt, user_prompt = get_estimation_prompt(structured_requirements, context)
	response = client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
	parsed = _safe_parse_json(response)
	if parsed is not None:
		return parsed

	logger.warning("Invalid JSON from model, retrying with correction prompt")
	correction_prompt = (
		"Your previous output was invalid JSON. "
		"Return ONLY valid JSON matching the required schema. "
		"Do NOT include any text before or after JSON."
	)
	response = client.generate(system_prompt=system_prompt, user_prompt=correction_prompt)
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
	cleaned = _strip_json_fences(text)
	if cleaned is None:
		return None
	try:
		return json.loads(cleaned)
	except json.JSONDecodeError:
		return None


def _truncate_context(context: str, max_chars: int) -> str:
	if len(context) <= max_chars:
		return context
	logger.warning("Truncated retrieved context from %s to %s characters", len(context), max_chars)
	return context[:max_chars].rstrip()


def _strip_json_fences(text: str) -> str | None:
	if not text:
		return None
	stripped = text.strip()
	if stripped.startswith("```"):
		stripped = stripped.strip("`")
		stripped = stripped.replace("json", "", 1).strip()

	start = stripped.find("{")
	end = stripped.rfind("}")
	if start == -1 or end == -1 or end <= start:
		return None
	return stripped[start : end + 1]
