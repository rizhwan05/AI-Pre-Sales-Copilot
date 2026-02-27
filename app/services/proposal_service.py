from __future__ import annotations

from typing import Dict, List


def generate_proposal(
	rfp_text: str,
	requirements: Dict[str, object],
	similar_projects: List[Dict[str, object]],
) -> Dict[str, object]:
	if not rfp_text or not rfp_text.strip():
		raise ValueError("rfp_text must be a non-empty string")

	return {
		"overview": requirements.get("summary") if requirements else rfp_text.strip()[:200],
		"approach": "Phased delivery with discovery, build, and rollout.",
		"references": [p.get("project_name") for p in similar_projects if p.get("project_name")],
	}
