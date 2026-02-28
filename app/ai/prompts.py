from __future__ import annotations

from typing import Dict, Tuple


def get_extraction_prompt(section_title: str, section_text: str) -> Tuple[str, str]:
	system_prompt = f"""You are an expert requirements analyst for pre-sales RFPs.
You MUST return ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.
Do NOT include triple backticks.
Do NOT include text before or after JSON.
Output MUST start with '{{' and end with '}}'.
If you output anything other than valid JSON, the response will be rejected.
Use only the provided input. No hallucinations.
"""
	user_prompt = f"""Extract requirements from the section below and return JSON with keys:
functional_requirements, non_functional_requirements, constraints, compliance_items, assumptions.
Each value must be a list of strings.

Schema example (use same keys and list-of-strings values):
{{
  "functional_requirements": [
    "Requirement 1",
    "Requirement 2"
  ],
  "non_functional_requirements": [
    "Requirement 1"
  ],
  "constraints": [
    "Constraint 1"
  ],
  "compliance_items": [
    "Compliance item 1"
  ],
  "assumptions": [
    "Assumption 1"
  ]
}}

SECTION TITLE: {section_title}
SECTION TEXT:
{section_text}
"""
	return system_prompt, user_prompt


def get_refinement_prompt(aggregated_requirements: Dict[str, object]) -> Tuple[str, str]:
	system_prompt = """You are a requirements editor. Return ONLY valid JSON.
Remove duplicates, merge similar items, standardize wording, and validate consistency.
Use only the provided input. No hallucinations.
"""
	user_prompt = f"""Refine the aggregated requirements below and return JSON with keys:
functional_requirements, non_functional_requirements, constraints, compliance_items, assumptions.
Each value must be a list of strings.

AGGREGATED REQUIREMENTS JSON:
{aggregated_requirements}
"""
	return system_prompt, user_prompt


def get_proposal_prompt(
	structured_requirements: Dict[str, object],
	retrieved_context: str,
) -> Tuple[str, str]:
	system_prompt = (
		"You are a pre-sales proposal writer. Return ONLY valid JSON. "
		"Use only the provided requirements and context. No hallucinations."
	)
	user_prompt = f"""Generate a proposal response in JSON with keys:
executive_summary, solution_approach, architecture_overview, differentiators.
Each value must be a string.

STRUCTURED REQUIREMENTS JSON:
{structured_requirements}

RETRIEVED CONTEXT:
{retrieved_context}
"""
	return system_prompt, user_prompt


def get_estimation_prompt(
	structured_requirements: Dict[str, object],
	retrieved_context: str,
) -> Tuple[str, str]:
	system_prompt = (
		"You are a delivery estimation specialist. Return ONLY valid JSON. "
		"Use only the provided requirements and context. No hallucinations."
	)
	user_prompt = f"""Generate an estimation response in JSON with keys:
team_composition, timeline_months, phase_breakdown, risk_adjustments.
team_composition and phase_breakdown must be lists of strings;
timeline_months must be a number; risk_adjustments must be a list of strings.

STRUCTURED REQUIREMENTS JSON:
{structured_requirements}

RETRIEVED CONTEXT:
{retrieved_context}
"""
	return system_prompt, user_prompt
