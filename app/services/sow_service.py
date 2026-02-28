from __future__ import annotations

import json
import logging
from typing import Dict, List

from pydantic import ValidationError

from app.ai.bedrock_client import BedrockClient
from app.ai.prompts import get_sow_index_prompt, get_sow_prompt, get_sow_section_prompt
from app.services.retrieval_service import retrieve_similar_projects
from app.services.sow_schema import SOWSchema
from llama_index.core.utils import get_tokenizer

logger = logging.getLogger(__name__)

SOW_TOP_K = 18
MAX_CONTEXT_CHARS = 12000


def generate_statement_of_work(structured_requirements: Dict[str, object]) -> Dict[str, object]:
	if not structured_requirements:
		raise ValueError("structured_requirements must be provided")

	query_text = _summarize_requirements(structured_requirements)
	retrieved = retrieve_similar_projects(query_text, document_type=None, top_k=SOW_TOP_K)
	context = _format_retrieved_context(retrieved)
	context = _truncate_context(context, MAX_CONTEXT_CHARS)
	token_count = _estimate_tokens(context)

	if not retrieved:
		logger.warning("SOW has no retrieved context; using RFP summary only")

	client = BedrockClient()
	llm_calls = 0

	index_prompt, index_user = get_sow_index_prompt(structured_requirements, context)
	index_user = (
		f"{index_user}\n\n"
		f"CONTEXT_STATUS: {'none' if not retrieved else 'available'}\n"
		"CONTEXT_CONFLICTS: prefer RFP requirements if conflicts are detected\n"
	)
	index_response = client.generate(system_prompt=index_prompt, user_prompt=index_user)
	llm_calls += 1
	index_data = _safe_parse_json(index_response) or {}
	sections = index_data.get("sections") if isinstance(index_data, dict) else None
	activities = index_data.get("activities") if isinstance(index_data, dict) else None
	if not isinstance(sections, list) or not sections:
		sections = _default_sections()
	if not isinstance(activities, list):
		activities = []

	merged = _empty_sow_document()
	sections_generated = 0
	for section_name in sections:
		section_prompt, section_user = get_sow_section_prompt(
			structured_requirements,
			context,
			section_name=str(section_name),
			activities=activities,
		)
		section_user = (
			f"{section_user}\n\n"
			f"CONTEXT_STATUS: {'none' if not retrieved else 'available'}\n"
			"CONTEXT_CONFLICTS: prefer RFP requirements if conflicts are detected\n"
		)
		section_response = client.generate(system_prompt=section_prompt, user_prompt=section_user)
		llm_calls += 1
		section_payload = _safe_parse_json(section_response)
		if section_payload is None:
			logger.warning("Invalid JSON for SOW section %s, retrying once", section_name)
			section_response = client.generate(system_prompt=section_prompt, user_prompt=section_user)
			llm_calls += 1
			section_payload = _safe_parse_json(section_response)
		if section_payload is None:
			logger.warning("Failed to generate SOW section %s; using defaults", section_name)
			continue
		_merge_section(merged, section_payload)
		sections_generated += 1

	logger.info(
		"SOW context: generation_mode=%s number_of_chunks=%s context_char_length=%s approximate_tokens=%s number_of_sections_generated=%s total_llm_calls=%s",
		"statement_of_work",
		len(retrieved),
		len(context),
		token_count,
		sections_generated,
		llm_calls,
	)

	try:
		validated = _validate_sow_dict(merged)
		if validated is not None:
			return validated
	except ValueError:
		pass

	logger.warning("Invalid JSON from model, retrying with correction prompt")
	correction_system, correction_user = get_sow_prompt(structured_requirements, context)
	correction_user = (
		f"{correction_user}\n\n"
		"Your previous response did not match schema. You MUST follow the schema exactly."
	)
	correction_response = client.generate(system_prompt=correction_system, user_prompt=correction_user)
	llm_calls += 1
	validated = _validate_sow_json(correction_response)
	return validated


def _validate_sow_json(text: str) -> Dict[str, object]:
	try:
		return SOWSchema.model_validate_json(text).model_dump()
	except ValidationError:
		cleaned = _strip_json_fences(text)
		if cleaned is None:
			raise ValueError("SOW JSON failed schema validation")
		try:
			return SOWSchema.model_validate_json(cleaned).model_dump()
		except ValidationError as exc:
			raise ValueError("SOW JSON failed schema validation") from exc


def _validate_sow_dict(payload: Dict[str, object]) -> Dict[str, object]:
	try:
		return SOWSchema.model_validate(payload).model_dump()
	except ValidationError as exc:
		raise ValueError("SOW JSON failed schema validation") from exc


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


def _safe_parse_json(text: str) -> Dict[str, object] | None:
	cleaned = _strip_json_fences(text)
	if cleaned is None:
		return None
	try:
		return json.loads(cleaned)
	except json.JSONDecodeError:
		return None


def _default_sections() -> List[str]:
	return [
		"title",
		"date",
		"version",
		"provider",
		"client",
		"executive_summary",
		"confidentiality",
		"business_value",
		"definitions",
		"scope_of_services",
		"service_delivery",
		"resource_profiles",
		"fees",
		"expenses",
		"change_management",
		"acceptance_and_approvals",
	]


def _empty_sow_document() -> Dict[str, object]:
	return {
		"document": {
			"title": "",
			"date": "",
			"version": "",
			"provider": "",
			"client": "",
			"executive_summary": {
				"purpose": "",
				"methodology": "",
				"key_domains": [],
				"art_of_possible": "",
			},
			"confidentiality": {"provisions": []},
			"business_value": {"value_opportunities": [], "alignment_with_goals": []},
			"definitions": {
				"Scope": "",
				"Deliverable": "",
				"Objectives": "",
				"Project Resource": "",
				"Success Criteria": "",
				"Exclusions": "",
			},
			"scope_of_services": {
				"activities": [],
				"deliverables_in_scope": [],
				"out_of_scope": [],
				"statement_of_understanding": [],
				"assumptions": [],
				"design_considerations": [],
				"customer_responsibilities": [],
				"project_resumption_charge": {"description": ""},
			},
			"service_delivery": {"delivery_methodology": ""},
			"resource_profiles": {"roles": [], "resource_allocation": ""},
			"fees": {"estimated_duration_weeks": 0, "invoicing_schedule": [], "notes": []},
			"expenses": {"reimbursed": []},
			"change_management": {"requirement": ""},
			"acceptance_and_approvals": {"terms": [], "signature_blocks": []},
		}
	}


def _merge_section(target: Dict[str, object], section_payload: Dict[str, object]) -> None:
	if not isinstance(section_payload, dict):
		return
	doc = target.get("document")
	if not isinstance(doc, dict):
		return
	for key, value in section_payload.items():
		if key in doc:
			doc[key] = value


def _estimate_tokens(text: str) -> int:
	if not text:
		return 0
	tokenizer = get_tokenizer()
	return len(tokenizer(text))
