from __future__ import annotations

from typing import Dict, List


def extract_requirements(rfp_text: str) -> Dict[str, object]:
	if not rfp_text or not rfp_text.strip():
		raise ValueError("rfp_text must be a non-empty string")

	sentences = [s.strip() for s in rfp_text.split(".") if s.strip()]
	key_points: List[str] = sentences[:5]
	return {
		"summary": sentences[0] if sentences else rfp_text.strip()[:200],
		"key_points": key_points,
		"source_length": len(rfp_text),
	}
