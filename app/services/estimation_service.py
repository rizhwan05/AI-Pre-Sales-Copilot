from __future__ import annotations

from typing import Dict


def estimate_effort(
	rfp_text: str,
	requirements: Dict[str, object],
	proposal: Dict[str, object],
) -> Dict[str, object]:
	if not rfp_text or not rfp_text.strip():
		raise ValueError("rfp_text must be a non-empty string")

	size = "M" if len(rfp_text) < 5000 else "L"
	return {
		"t_shirt_size": size,
		"confidence": 0.3,
		"notes": "Preliminary estimate based on RFP length only.",
	}
