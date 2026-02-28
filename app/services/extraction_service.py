from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple

from app.ai.bedrock_client import BedrockClient
from app.ai.prompts import get_extraction_prompt, get_refinement_prompt

logger = logging.getLogger(__name__)

_HEADING_REGEX = re.compile(r"^(?:(?:\d+)(?:\.\d+)*\.?\s+)?[A-Z][A-Z0-9\s\-\/]{3,}$")


def extract_requirements(rfp_text: str) -> Dict[str, object]:
	if not rfp_text or not rfp_text.strip():
		raise ValueError("rfp_text must be a non-empty string")

	clean_text = _clean_text(rfp_text)
	sections = _split_into_sections(clean_text)
	if not sections:
		sections = [("RFP", clean_text)]

	client = BedrockClient()
	aggregated = _init_requirements()

	for title, text in sections:
		logger.info("Extracting requirements from section: %s", title)
		system_prompt, user_prompt = get_extraction_prompt(title, text)
		section_data = _invoke_with_retry(client, system_prompt, user_prompt)
		_merge_requirements(aggregated, section_data)

	logger.info("Refining aggregated requirements")
	ref_system, ref_user = get_refinement_prompt(aggregated)
	refined = _invoke_with_retry(client, ref_system, ref_user)
	return refined


def _clean_text(text: str) -> str:
	return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
	lines = text.splitlines()
	sections: List[Tuple[str, str]] = []
	current_title = "Introduction"
	current_lines: List[str] = []

	for line in lines:
		if _HEADING_REGEX.match(line.strip()):
			if current_lines:
				sections.append((current_title, "\n".join(current_lines).strip()))
			current_title = line.strip()
			current_lines = []
		else:
			current_lines.append(line)

	if current_lines:
		sections.append((current_title, "\n".join(current_lines).strip()))
	return sections


def _invoke_with_retry(client: BedrockClient, system_prompt: str, user_prompt: str) -> Dict[str, object]:
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
	if parsed is not None:
		return parsed

	raise ValueError("LLM returned invalid JSON after retry")


def _safe_parse_json(text: str) -> Dict[str, object] | None:
	cleaned = _strip_json_fences(text)
	if cleaned is None:
		return None
	try:
		return json.loads(cleaned)
	except json.JSONDecodeError:
		return None


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


def _init_requirements() -> Dict[str, object]:
	return {
		"functional_requirements": [],
		"non_functional_requirements": [],
		"constraints": [],
		"compliance_items": [],
		"assumptions": [],
	}


def _merge_requirements(target: Dict[str, object], section_data: Dict[str, object]) -> None:
	for key in target.keys():
		values = section_data.get(key, [])
		if isinstance(values, list):
			target[key].extend([str(value) for value in values if value])
