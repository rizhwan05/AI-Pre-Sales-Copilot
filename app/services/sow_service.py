from __future__ import annotations

import json
import logging
import time
from typing import Dict, List

from app.ai.bedrock_client import BedrockClient
from app.ai.prompts import get_sow_index_prompt, get_sow_prompt, get_sow_section_prompt
from app.services.retrieval_service import retrieve_similar_projects
from llama_index.core.utils import get_tokenizer

logger = logging.getLogger(__name__)

SOW_TOP_K = 12
MAX_CONTEXT_CHARS = 12000
SECTION_CONTEXT_CHARS = 8000
MAX_SECTIONS = 10


def generate_statement_of_work(structured_requirements: Dict[str, object]) -> Dict[str, object]:
	if not structured_requirements:
		raise ValueError("structured_requirements must be provided")

	start_total = time.perf_counter()
	total_retries = 0

	query_text = _summarize_requirements(structured_requirements)
	logger.info("SOW_RETRIEVAL_START: timestamp=%s", time.time())
	retrieval_start = time.perf_counter()
	retrieved = retrieve_similar_projects(query_text, document_type=None, top_k=SOW_TOP_K)
	retrieval_duration_ms = int((time.perf_counter() - retrieval_start) * 1000)
	context = _format_retrieved_context(retrieved)
	context = _truncate_context(context, MAX_CONTEXT_CHARS)
	token_count = _estimate_tokens(context)
	logger.info(
		"SOW_RETRIEVAL_END: number_of_chunks=%s context_char_length=%s approximate_token_estimate=%s retrieval_duration_ms=%s",
		len(retrieved),
		len(context),
		token_count,
		retrieval_duration_ms,
	)

	if not retrieved:
		logger.warning("SOW has no retrieved context; using RFP summary only")

	client = BedrockClient()
	llm_calls = 0

	logger.info("SOW_INDEX_START: timestamp=%s", time.time())
	index_start = time.perf_counter()
	index_prompt, index_user = get_sow_index_prompt(structured_requirements, context)
	index_user = (
		f"{index_user}\n\n"
		f"CONTEXT_STATUS: {'none' if not retrieved else 'available'}\n"
		"CONTEXT_CONFLICTS: prefer RFP requirements if conflicts are detected\n"
	)
	try:
		index_response = client.generate(system_prompt=index_prompt, user_prompt=index_user)
	except RuntimeError as exc:
		raise RuntimeError("SOW index generation failed") from exc
	llm_calls += 1
	index_duration_ms = int((time.perf_counter() - index_start) * 1000)
	index_data = _safe_parse_json(index_response) or {}
	index_json_size = len(index_response) if index_response else 0
	logger.info(
		"SOW_INDEX_END: duration_ms=%s retry_count=%s index_json_size=%s",
		index_duration_ms,
		0,
		index_json_size,
	)
	sections = index_data.get("sections") if isinstance(index_data, dict) else None
	activities = index_data.get("activities") if isinstance(index_data, dict) else None
	if not isinstance(sections, list) or not sections:
		sections = _default_sections()
	if not isinstance(activities, list):
		activities = []
	if len(sections) > MAX_SECTIONS:
		logger.info("SOW sections truncated: original=%s capped=%s", len(sections), MAX_SECTIONS)
		sections = sections[:MAX_SECTIONS]

	merged = _empty_sow_document()
	sections_generated = 0
	sections_expected = len(sections)
	for section_name in sections:
		section_start = time.perf_counter()
		logger.info("SOW_SECTION_START: section_name=%s timestamp=%s", section_name, time.time())
		section_context = _truncate_context(context, SECTION_CONTEXT_CHARS)
		section_prompt, section_user = get_sow_section_prompt(
			structured_requirements,
			section_context,
			section_name=str(section_name),
			activities=activities,
		)
		section_user = (
			f"{section_user}\n\n"
			f"CONTEXT_STATUS: {'none' if not retrieved else 'available'}\n"
			"CONTEXT_CONFLICTS: prefer RFP requirements if conflicts are detected\n"
		)
		section_retry_count = 0
		section_call_start = time.perf_counter()
		try:
			section_response = client.generate(system_prompt=section_prompt, user_prompt=section_user)
		except RuntimeError as exc:
			raise RuntimeError("SOW section generation failed") from exc
		section_call_duration_ms = int((time.perf_counter() - section_call_start) * 1000)
		logger.info(
			"SOW_SECTION_CALL: section_name=%s duration_ms=%s retry_count=%s",
			section_name,
			section_call_duration_ms,
			section_retry_count,
		)
		llm_calls += 1
		section_payload = _safe_parse_json(section_response)
		if section_payload is None:
			logger.warning("Invalid JSON for SOW section %s, retrying once", section_name)
			section_retry_count = 1
			section_call_start = time.perf_counter()
			try:
				section_response = client.generate(system_prompt=section_prompt, user_prompt=section_user)
			except RuntimeError as exc:
				raise RuntimeError("SOW section generation failed") from exc
			section_call_duration_ms = int((time.perf_counter() - section_call_start) * 1000)
			logger.info(
				"SOW_SECTION_CALL: section_name=%s duration_ms=%s retry_count=%s",
				section_name,
				section_call_duration_ms,
				section_retry_count,
			)
			llm_calls += 1
			total_retries += 1
			section_payload = _safe_parse_json(section_response)
		if section_payload is None:
			logger.warning("Failed to generate SOW section %s; using defaults", section_name)
			section_duration_ms = int((time.perf_counter() - section_start) * 1000)
			logger.info(
				"SOW_SECTION_END: section_name=%s duration_ms=%s retry_count=%s output_char_length=%s success_or_skipped=%s",
				section_name,
				section_duration_ms,
				section_retry_count,
				len(section_response) if section_response else 0,
				"skipped",
			)
			continue
		_merge_section(merged, section_payload)
		sections_generated += 1
		section_duration_ms = int((time.perf_counter() - section_start) * 1000)
		logger.info(
			"SOW_SECTION_END: section_name=%s duration_ms=%s retry_count=%s output_char_length=%s success_or_skipped=%s",
			section_name,
			section_duration_ms,
			section_retry_count,
			len(section_response) if section_response else 0,
			"success",
		)

	merged_json_length = len(json.dumps(merged))
	missing_sections_count = max(sections_expected - sections_generated, 0)
	logger.info(
		"SOW_MERGE_END: total_sections_expected=%s total_sections_generated=%s missing_sections_count=%s merged_json_char_length=%s",
		sections_expected,
		sections_generated,
		missing_sections_count,
		merged_json_length,
	)

	logger.info(
		"SOW context: generation_mode=%s number_of_chunks=%s context_char_length=%s approximate_tokens=%s number_of_sections_generated=%s total_llm_calls=%s",
		"statement_of_work",
		len(retrieved),
		len(context),
		token_count,
		sections_generated,
		llm_calls,
	)

	logger.info("SOW_VALIDATION_START")
	validation_start = time.perf_counter()
	try:
		validated = _validate_sow_dict(merged)
		if validated is not None:
			validation_duration_ms = int((time.perf_counter() - validation_start) * 1000)
			logger.info(
				"SOW_VALIDATION_END: validation_duration_ms=%s success_or_failure=%s",
				validation_duration_ms,
				"success",
			)
			total_duration_ms = int((time.perf_counter() - start_total) * 1000)
			logger.info(
				"SOW_TOTAL: total_duration_ms=%s total_llm_calls=%s total_retries=%s final_json_char_length=%s",
				total_duration_ms,
				llm_calls,
				total_retries,
				len(json.dumps(validated)),
			)
			return validated
	except ValueError:
		validation_duration_ms = int((time.perf_counter() - validation_start) * 1000)
		logger.info(
			"SOW_VALIDATION_END: validation_duration_ms=%s success_or_failure=%s",
			validation_duration_ms,
			"failure",
		)
		pass

	logger.warning("Invalid JSON from model, retrying with correction prompt")
	correction_system, correction_user = get_sow_prompt(structured_requirements, context)
	correction_user = (
		f"{correction_user}\n\n"
		"Your previous response did not match schema. You MUST follow the schema exactly."
	)
	correction_call_start = time.perf_counter()
	try:
		correction_response = client.generate(system_prompt=correction_system, user_prompt=correction_user)
	except RuntimeError as exc:
		raise RuntimeError("SOW correction generation failed") from exc
	correction_call_duration_ms = int((time.perf_counter() - correction_call_start) * 1000)
	logger.info(
		"SOW_CORRECTION_CALL: duration_ms=%s retry_count=%s",
		correction_call_duration_ms,
		1,
	)
	llm_calls += 1
	total_retries += 1
	validated = _validate_sow_json(correction_response)
	validation_duration_ms = int((time.perf_counter() - validation_start) * 1000)
	logger.info(
		"SOW_VALIDATION_END: validation_duration_ms=%s success_or_failure=%s",
		validation_duration_ms,
		"success" if validated else "failure",
	)
	total_duration_ms = int((time.perf_counter() - start_total) * 1000)
	logger.info(
		"SOW_TOTAL: total_duration_ms=%s total_llm_calls=%s total_retries=%s final_json_char_length=%s",
		total_duration_ms,
		llm_calls,
		total_retries,
		len(json.dumps(validated)),
	)
	return validated


def _validate_sow_json(text: str) -> Dict[str, object]:
	cleaned = _strip_json_fences(text)
	if cleaned:
		parsed = _safe_parse_json(cleaned)
		if parsed is not None:
			return parsed
		repaired = _repair_json(cleaned)
		if repaired:
			parsed = _safe_parse_json(repaired)
			if parsed is not None:
				logger.warning("SOW JSON repaired after truncation")
				return parsed
	logger.warning("SOW JSON parsing failed; falling back to empty document")
	return _empty_sow_document()


def _validate_sow_dict(payload: Dict[str, object]) -> Dict[str, object]:
	if isinstance(payload, dict):
		return payload
	logger.warning("SOW payload was not a dict; falling back to empty document")
	return _empty_sow_document()


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


def _repair_json(text: str) -> str | None:
	if not text:
		return None
	stripped = text.strip()
	start = stripped.find("{")
	if start == -1:
		return None
	segment = stripped[start:]
	stack: List[str] = []
	in_string = False
	escape = False
	for ch in segment:
		if in_string:
			if escape:
				escape = False
				continue
			if ch == "\\":
				escape = True
				continue
			if ch == '"':
				in_string = False
			continue
		if ch == '"':
			in_string = True
			continue
		if ch == "{":
			stack.append("}")
			continue
		if ch == "[":
			stack.append("]")
			continue
		if ch in ("}", "]"):
			if stack and ch == stack[-1]:
				stack.pop()
			continue
	if in_string:
		return None
	if stack:
		segment = segment + "".join(reversed(stack))
	return segment


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
