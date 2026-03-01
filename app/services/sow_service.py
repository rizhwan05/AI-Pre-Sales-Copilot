from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

from app.ai.bedrock_client import BedrockClient
from app.ai.prompts import get_statement_of_work_prompt
from app.services.retrieval_service import retrieve_similar_projects

logger = logging.getLogger(__name__)

SOW_TOP_K = 12
SOW_CONTEXT_MAX_CHARS = 15000
SOW_MAX_TOKENS = 6000
SOW_MAX_SCHEMA_RETRIES = 2
SOW_MAX_MODEL_RETRIES = 2


def generate_statement_of_work(structured_requirements: Dict[str, object]) -> Dict[str, object]:
	if not structured_requirements:
		raise ValueError("structured_requirements must be provided")

	template = _build_sow_template()
	query_text = _summarize_requirements(structured_requirements)
	retrieved = retrieve_similar_projects(query_text, document_type=None, top_k=SOW_TOP_K)
	context = _format_retrieved_context(retrieved)
	context = _truncate_context(context, SOW_CONTEXT_MAX_CHARS)
	logger.info(
		"SOW LLM context: chunks=%s chars=%s empty=%s generation_mode=%s",
		len(retrieved),
		len(context),
		not context.strip(),
		"sow",
	)
	if not context.strip():
		logger.warning("SOW LLM answering from RFP structured summary only")

	client = BedrockClient()
	system_prompt, user_prompt = get_statement_of_work_prompt(structured_requirements, context)
	template_block = json.dumps(template, ensure_ascii=True)
	user_prompt = (
		f"{user_prompt}\n\n"
		"JSON TEMPLATE (use exactly; fill content fields and any empty header values only):\n"
		f"{template_block}"
	)
	validation_error = None
	for attempt in range(SOW_MAX_SCHEMA_RETRIES + 1):
		if attempt == 0:
			prompt = user_prompt
		else:
			prompt = (
				"Your previous output was invalid JSON. "
				"Return ONLY valid JSON matching the required schema. "
				"Do NOT include any text before or after JSON. "
				f"Validation error: {validation_error or 'Output did not match required schema'}"
			)

		response = _generate_with_retry(client, system_prompt, prompt)
		parsed = _safe_parse_json(response)
		if parsed is None:
			validation_error = "Invalid JSON"
			logger.warning("Invalid JSON from model on attempt %s", attempt + 1)
			continue

		merged = _merge_with_template(parsed, template)
		valid, error = _validate_sow_output(merged)
		if valid:
			return merged
		validation_error = error
		logger.warning("Invalid SOW schema on attempt %s: %s", attempt + 1, error)

	logger.error("SOW generation failed after retries; returning template fallback")
	return _fallback_sow(template)


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
		return "No similar projects found."

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
		return _extract_first_json_object(cleaned)


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


def _extract_first_json_object(text: str) -> Dict[str, object] | None:
	start = text.find("{")
	if start == -1:
		return None
	for idx in range(start + 1, len(text)):
		if text[idx] != "}":
			continue
		candidate = text[start : idx + 1]
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, dict):
				return parsed
		except json.JSONDecodeError:
			continue
	return None


def _build_sow_template() -> Dict[str, object]:
	return {
		"document": {
			"title": "",
			"date": "",
			"version": "",
			"sections": [
				{"title": "Executive Summary", "content": ""},
				{"title": "Statement of Confidentiality", "content": ""},
				{"title": "Understanding of Business Value and Goals", "content": ""},
				{"title": "Project Definitions", "content": ""},
				{"title": "Proposed Scope of Services", "content": ""},
				{"title": "Service Delivery Approach", "content": ""},
				{"title": "Resource Profiles", "content": ""},
				{"title": "Schedule of Fees", "content": ""},
				{"title": "Expenses", "content": ""},
				{"title": "Change Management", "content": ""},
				{"title": "Acceptance and Approvals", "content": ""},
			],
		}
	}


def _merge_with_template(parsed: Dict[str, object], template: Dict[str, object]) -> Dict[str, object]:
	merged = json.loads(json.dumps(template))
	document = parsed.get("document") if isinstance(parsed, dict) else None
	if not isinstance(document, dict):
		return merged

	for key in ("title", "date", "version"):
		value = document.get(key)
		if isinstance(value, str) and value.strip() and not merged["document"].get(key):
			merged["document"][key] = value.strip()

	sections = document.get("sections")
	if isinstance(sections, list):
		section_map = {}
		for item in sections:
			if not isinstance(item, dict):
				continue
			title = item.get("title")
			content = item.get("content")
			if isinstance(title, str) and isinstance(content, str):
				section_map[title.strip()] = content.strip()

		for item in merged["document"]["sections"]:
			title = item.get("title")
			if title in section_map and section_map[title]:
				item["content"] = section_map[title]

	return merged


def _validate_sow_output(payload: Dict[str, object]) -> Tuple[bool, str | None]:
	if not isinstance(payload, dict):
		return False, "Root must be an object"
	if "document" not in payload or not isinstance(payload["document"], dict):
		return False, "Missing or invalid document"
	document = payload["document"]
	for key in ("title", "date", "version", "sections"):
		if key not in document:
			return False, f"Missing document.{key}"
	if not isinstance(document["sections"], list) or not document["sections"]:
		return False, "document.sections must be a non-empty list"
	template_titles = [item.get("title") for item in _build_sow_template()["document"]["sections"]]
	section_titles = [item.get("title") for item in document["sections"]]
	if section_titles != template_titles:
		return False, "document.sections titles must match the fixed template order"
	for section in document["sections"]:
		if not isinstance(section, dict):
			return False, "Each section must be an object"
		if not isinstance(section.get("title"), str):
			return False, "Each section.title must be a string"
		if not isinstance(section.get("content"), str):
			return False, "Each section.content must be a string"
	return True, None


def _generate_with_retry(
	client: BedrockClient,
	system_prompt: str,
	user_prompt: str,
) -> str:
	last_error: Exception | None = None
	for attempt in range(SOW_MAX_MODEL_RETRIES + 1):
		try:
			return client.generate(
				system_prompt=system_prompt,
				user_prompt=user_prompt,
				max_tokens=SOW_MAX_TOKENS,
			)
		except RuntimeError as exc:
			last_error = exc
			logger.warning("SOW LLM call failed on attempt %s: %s", attempt + 1, exc)
			continue
	if last_error:
		logger.error("SOW LLM call failed after retries: %s", last_error)
	return ""


def _fallback_sow(template: Dict[str, object]) -> Dict[str, object]:
	merged = json.loads(json.dumps(template))
	sections = merged.get("document", {}).get("sections", [])
	if isinstance(sections, list) and sections:
		sections[0]["content"] = (
			"SOW content could not be generated due to a temporary system error. "
			"Please retry."
		)
	return merged
