from __future__ import annotations

import logging
import os
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "anthropic.claude-opus-4-6"


class BedrockClient:
	def __init__(self, model_id: str = DEFAULT_MODEL_ID, region: Optional[str] = None) -> None:
		self._model_id = model_id
		self._region = region or "us-east-1"
		_ensure_bedrock_bearer_token()
		config = Config(connect_timeout=30, read_timeout=180, retries={"max_attempts": 3, "mode": "standard"})
		self._client = boto3.client(
			service_name="bedrock-runtime",
			region_name=self._region,
			config=config,
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
		system_prompt = system_prompt or ""

		try:
			request_kwargs = {
				"modelId": self._model_id,
				"messages": [
					{
						"role": "user",
						"content": [{"text": user_prompt}],
					}
				],
				"inferenceConfig": {
					"temperature": float(temperature),
					"maxTokens": int(max_tokens),
				},
			}
			if system_prompt.strip():
				request_kwargs["system"] = [{"text": system_prompt}]

			response = self._client.converse(**request_kwargs)
			text = _extract_converse_text(response)
			if not text:
				raise RuntimeError("Empty response from Bedrock model")
			return text
		except self._client.exceptions.ValidationException as exc:
			logger.exception("Bedrock request validation failed")
			raise RuntimeError("Bedrock request validation failed") from exc
		except self._client.exceptions.ThrottlingException as exc:
			logger.exception("Bedrock request throttled")
			raise RuntimeError("Bedrock request throttled") from exc
		except self._client.exceptions.AccessDeniedException as exc:
			logger.exception("Bedrock access denied")
			raise RuntimeError("Bedrock access denied") from exc
		except self._client.exceptions.ResourceNotFoundException as exc:
			logger.exception("Bedrock model not found")
			raise RuntimeError("Bedrock model not found") from exc
		except self._client.exceptions.ServiceQuotaExceededException as exc:
			logger.exception("Bedrock service quota exceeded")
			raise RuntimeError("Bedrock service quota exceeded") from exc
		except self._client.exceptions.InternalServerException as exc:
			logger.exception("Bedrock internal server error")
			raise RuntimeError("Bedrock internal server error") from exc
		except (BotoCoreError, ClientError) as exc:
			logger.exception("Bedrock request failed")
			raise RuntimeError("Bedrock request failed") from exc
		except (ValueError, KeyError, TypeError, AttributeError) as exc:
			logger.exception("Unexpected Bedrock response format")
			raise RuntimeError("Unexpected Bedrock response format") from exc


def _ensure_bedrock_bearer_token() -> None:
	if not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
		raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK not configured")


def _extract_converse_text(payload: dict) -> str:
	output = payload.get("output")
	if not isinstance(output, dict):
		return ""
	message = output.get("message")
	if not isinstance(message, dict):
		return ""
	content = message.get("content")
	if isinstance(content, list):
		parts = [item.get("text", "") for item in content if isinstance(item, dict)]
		return "".join(parts).strip()
	return ""
