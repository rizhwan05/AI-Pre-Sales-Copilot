from __future__ import annotations

from typing import Dict, Tuple


def get_extraction_prompt(section_title: str, section_text: str) -> Tuple[str, str]:
	system_prompt = (
		"You are an expert requirements analyst for pre-sales RFPs. "
		"Return ONLY valid JSON. Do not add commentary. "
		"Use only the provided input. No hallucinations."
	)
	user_prompt = (
		"Extract requirements from the section below and return JSON with keys: "
		"functional_requirements, non_functional_requirements, constraints, compliance_items, assumptions. "
		"Each value must be a list of strings.\n\n"
		f"SECTION TITLE: {section_title}\n"
		f"SECTION TEXT:\n{section_text}\n"
	)
	return system_prompt, user_prompt


def get_refinement_prompt(aggregated_requirements: Dict[str, object]) -> Tuple[str, str]:
	system_prompt = (
		"You are a requirements editor. Return ONLY valid JSON. "
		"Remove duplicates, merge similar items, standardize wording, and validate consistency. "
		"Use only the provided input. No hallucinations."
	)
	user_prompt = (
		"Refine the aggregated requirements below and return JSON with keys: "
		"functional_requirements, non_functional_requirements, constraints, compliance_items, assumptions. "
		"Each value must be a list of strings.\n\n"
		f"AGGREGATED REQUIREMENTS JSON:\n{aggregated_requirements}\n"
	)
	return system_prompt, user_prompt


def get_proposal_prompt(
	structured_requirements: Dict[str, object],
	retrieved_context: str,
) -> Tuple[str, str]:
	system_prompt = (
		"You are a pre-sales proposal writer. Return ONLY valid JSON. "
		"Use only the provided requirements and context. No hallucinations."
	)
	user_prompt = (
		"Generate a proposal response in JSON with keys: "
		"executive_summary, solution_approach, architecture_overview, differentiators. "
		"Each value must be a string.\n\n"
		f"STRUCTURED REQUIREMENTS JSON:\n{structured_requirements}\n\n"
		f"RETRIEVED CONTEXT:\n{retrieved_context}\n"
	)
	return system_prompt, user_prompt


def get_estimation_prompt(
	structured_requirements: Dict[str, object],
	retrieved_context: str,
) -> Tuple[str, str]:
	system_prompt = (
		"You are a delivery estimation specialist. Return ONLY valid JSON. "
		"Use only the provided requirements and context. No hallucinations."
	)
	user_prompt = (
		"Generate an estimation response in JSON with keys: "
		"team_composition, timeline_months, phase_breakdown, risk_adjustments. "
		"team_composition and phase_breakdown must be lists of strings; "
		"timeline_months must be a number; risk_adjustments must be a list of strings.\n\n"
		f"STRUCTURED REQUIREMENTS JSON:\n{structured_requirements}\n\n"
		f"RETRIEVED CONTEXT:\n{retrieved_context}\n"
	)
	return system_prompt, user_prompt
