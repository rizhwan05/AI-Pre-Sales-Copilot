from __future__ import annotations

import json
import logging
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "anthropic.claude-opus-4-6"


class BedrockClient:
	def __init__(self, model_id: str = DEFAULT_MODEL_ID, region: Optional[str] = None) -> None:
		self._model_id = model_id
		self._region = region or "us-east-1"
		self._client = boto3.client(
			service_name="bedrock-runtime",
			region_name=self._region,
		)
		logger.info("Bedrock client initialized using bearer token authentication")

	def generate(
		self,
		system_prompt: str,
		user_prompt: str,
		temperature: float = 0.2,
		max_tokens: int = 4000,
	) -> str:
		if not user_prompt or not user_prompt.strip():
			raise ValueError("user_prompt must be a non-empty string")
		if not system_prompt:
			system_prompt = ""

		payload = {
			"anthropic_version": "bedrock-2023-05-31",
			"system": system_prompt,
			"messages": [
				{
					"role": "user",
					"content": [{"type": "text", "text": user_prompt}],
				}
			],
			"temperature": float(temperature),
			"max_tokens": int(max_tokens),
		}

		try:
			response = self._client.invoke_model(
				modelId=self._model_id,
				contentType="application/json",
				accept="application/json",
				body=json.dumps(payload),
			)
			body = json.loads(response.get("body").read())
			text = _extract_text(body)
			if not text:
				raise RuntimeError("Empty response from Bedrock model")
			return text
		except (BotoCoreError, ClientError) as exc:
			logger.exception("Bedrock request failed")
			raise RuntimeError("Bedrock request failed") from exc
		except (ValueError, KeyError, TypeError) as exc:
			logger.exception("Unexpected Bedrock response format")
			raise RuntimeError("Unexpected Bedrock response format") from exc


def _extract_text(payload: dict) -> str:
	content = payload.get("content")
	if isinstance(content, list):
		parts = [item.get("text", "") for item in content if isinstance(item, dict)]
		return "".join(parts).strip()
	return ""
