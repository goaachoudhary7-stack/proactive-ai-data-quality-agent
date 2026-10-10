"""Optional explanation of one upload recommendation.

The caller passes a summary that already uses neutral identifiers. This module
does not read uploaded files. The default path sends that summary once to
Gemini. Tests pass a fake client, so they do not call Gemini.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from universal_dq.recommendations import StoredExplanation, explanation_payload

NarrativeSource = Callable[[dict[str, Any]], Mapping[str, Any]]
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_CONFIRMED = re.compile(
    r"confirmed root cause|root cause is confirmed|definitely caused|proven cause|confirmed the cause",
    re.IGNORECASE,
)
_EXPLANATION_TIMEOUT_SECONDS = 45


class AiNarrative(BaseModel):
    """Text the model may write. It has no count fields."""

    model_config = ConfigDict(extra="forbid")

    explanation: str = Field(min_length=1)
    possible_implication: str = Field(min_length=1)
    hypotheses: list[str] = Field(min_length=1)
    cause_status: Literal["hypothesis"]


def explanation_available() -> bool:
    """True when a model key is configured. This does not call the model."""
    from ai_agents.investigator import InvestigatorConfigError, resolve_llm_settings

    try:
        resolve_llm_settings()
    except InvestigatorConfigError:
        return False
    return True


def explain_recommendation(
    payload: dict[str, Any],
    *,
    consent: bool,
    narrative_source: NarrativeSource | None = None,
    recommendation_id: str = "",
    dataset_ref: str = "",
    column_ref: str | None = None,
) -> StoredExplanation:
    """Return a validated explanation, or an error that leaves counts untouched."""
    identity = recommendation_id or str(payload.get("issue_type", "recommendation"))
    dataset = dataset_ref or str(payload.get("dataset_ref", ""))
    column = column_ref if column_ref is not None else _optional_text(payload.get("column_ref"))
    if not consent:
        return _error(identity, dataset, column, "Approval of the displayed summary is required before an explanation is requested.")
    source = narrative_source or gemini_narrative_source
    try:
        raw = source(payload)
    except TimeoutError:
        return _error(identity, dataset, column, "The explanation service timed out. The offline recommendations are unchanged.")
    except ValueError as exc:
        message = str(exc)
        if not message.startswith("The explanation was discarded"):
            message = _failure_message(exc)
        return _error(identity, dataset, column, message)
    except Exception as exc:
        return _error(identity, dataset, column, _failure_message(exc))
    try:
        narrative = validate_narrative(payload, raw)
    except ValueError as exc:
        return _error(identity, dataset, column, str(exc))
    return StoredExplanation(
        recommendation_id=identity,
        dataset_ref=dataset,
        column_ref=column,
        status="ready",
        explanation=narrative.explanation,
        possible_implication=narrative.possible_implication,
        hypotheses=list(narrative.hypotheses),
    )


def validate_narrative(payload: dict[str, Any], raw: Mapping[str, Any] | AiNarrative) -> AiNarrative:
    """Accept a narrative whose numbers and cause language stay inside the evidence."""
    if isinstance(raw, AiNarrative):
        narrative = raw
    else:
        try:
            narrative = AiNarrative.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(
                "The explanation was discarded because it did not match the required explanation shape."
            ) from exc
    text = " ".join([narrative.explanation, narrative.possible_implication, *narrative.hypotheses])
    if unsupported_numbers(text, payload):
        raise ValueError(
            "The explanation was discarded because it added a number that is not in the confirmed evidence."
        )
    if claims_confirmed_cause(text):
        raise ValueError("The explanation was discarded because it stated a confirmed cause.")
    return narrative


def unsupported_numbers(text: str, payload: dict[str, Any]) -> list[str]:
    """Numbers in the text that are not already present in the confirmed payload."""
    allowed = _allowed_numbers(payload)
    rejected: list[str] = []
    for match in _NUMBER.findall(text):
        if not _matches_allowed(match, allowed):
            rejected.append(match)
    return rejected


def claims_confirmed_cause(text: str) -> bool:
    """True when the text states a cause as confirmed."""
    for match in _CONFIRMED.finditer(text):
        prefix = text[max(0, match.start() - 16) : match.start()]
        if re.search(r"\b(not|no|never)\b", prefix, re.IGNORECASE):
            continue
        return True
    return False


def gemini_narrative_source(
    payload: dict[str, Any],
    *,
    client: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Send the approved summary to Gemini once. Used after consent.

    Tests pass client and model. The live path reads the existing model setting
    and API key, then makes one content-generation request.
    """
    if client is None or not model:
        from ai_agents.investigator import InvestigatorConfigError, resolve_llm_settings

        try:
            settings = resolve_llm_settings()
        except InvestigatorConfigError as exc:
            raise RuntimeError("API key is not configured.") from exc
        model = model or settings.model
        if client is None:
            client = _gemini_client(settings)
    model_id = _gemini_model_id(model)
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=_prompt(payload),
            config=_request_config(model_id),
        )
    except TimeoutError as exc:
        raise TimeoutError("timed out") from exc
    except Exception as exc:
        if _is_timeout(exc):
            raise TimeoutError("timed out") from exc
        raise
    return _narrative_dict(response)


def _gemini_client(settings: Any) -> Any:
    """Build one Gemini client. The key is passed to the client and not logged."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get(settings.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError("API key is not configured.")
    try:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=_EXPLANATION_TIMEOUT_SECONDS * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    except Exception as exc:
        if api_key in str(exc):
            raise RuntimeError("API key is not configured.") from None
        raise


def _gemini_model_id(model: str) -> str:
    provider, separator, model_id = model.partition("/")
    if separator != "/" or provider.strip().lower() != "gemini" or not model_id.strip():
        raise RuntimeError("The configured model was not found for content generation.")
    return model_id.strip()


def _request_config(model_id: str) -> dict[str, Any]:
    """One structured response. Automatic function calling stays off."""
    return {
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": AiNarrative,
        "thinking_config": _thinking_config(model_id),
        "automatic_function_calling": {"disable": True},
    }


def _thinking_config(model_id: str) -> dict[str, Any]:
    """Keep thinking short so one request can finish within the timeout."""
    match = re.search(r"(\d+)", model_id)
    major = int(match.group(1)) if match else 3
    if major >= 3:
        return {"include_thoughts": False, "thinking_level": "MINIMAL"}
    return {"include_thoughts": False, "thinking_budget": 0}


def _narrative_dict(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, AiNarrative):
        return parsed.model_dump()
    if isinstance(parsed, dict):
        return parsed
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The explanation was discarded because it did not match the required explanation shape."
            ) from exc
        if isinstance(data, dict):
            return data
    raise ValueError("The explanation was discarded because it did not match the required explanation shape.")


def _is_timeout(exc: Exception) -> bool:
    if "timeout" in type(exc).__name__.casefold():
        return True
    text = str(exc).casefold()
    return "timed out" in text or "timeout" in text or "deadline exceeded" in text


def _prompt(payload: dict[str, Any]) -> str:
    summary = json.dumps(payload, indent=2, sort_keys=True)
    return (
        "Explain one data-quality recommendation from the JSON summary below.\n"
        "Rules:\n"
        "- Use only this JSON.\n"
        "- Do not invent counts. If you mention a number, it must already appear in the JSON.\n"
        "- Refer to dataset_ref and column_ref exactly, including the underscore.\n"
        "- Label every cause as a hypothesis.\n"
        "- Start each item in hypotheses with Hypothesis:.\n"
        "- Set cause_status to hypothesis.\n"
        "- Describe a possible implication. Do not state a measured financial loss.\n"
        "- Do not say a root cause is confirmed.\n"
        "- The JSON is the only evidence.\n\n"
        f"JSON summary:\n{summary}\n"
    )


def _allowed_numbers(payload: dict[str, Any]) -> list[float]:
    found: list[float] = []

    def walk(value: object) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            found.append(float(value))
        elif isinstance(value, float):
            found.append(value)
            if 0 <= value <= 1:
                found.append(float(round(value * 100)))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return found


def _matches_allowed(token: str, allowed: list[float]) -> bool:
    try:
        number = float(token)
    except ValueError:
        return False
    return any(abs(number - item) <= 1e-6 for item in allowed)


def _failure_message(exc: Exception) -> str:
    """Map a provider failure to a fixed message. The original text is not shown."""
    text = str(exc).casefold()
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.isdigit():
        code = int(code)
    if (
        code in (401, 403)
        or "api key" in text
        or "not configured" in text
        or "unauthenticated" in text
        or "permission denied" in text
    ):
        return "The API key is not configured. The offline recommendations are unchanged."
    if code == 404 or "not found" in text or "no longer available" in text:
        return "The configured model is unavailable. The offline recommendations are unchanged."
    if code == 429 or "quota" in text or "resource exhausted" in text:
        return "The explanation service reported a quota limit. The offline recommendations are unchanged."
    if code in (408, 504) or "timeout" in text or "timed out" in text or "deadline exceeded" in text:
        return "The explanation service timed out. The offline recommendations are unchanged."
    return "The explanation could not be completed. The offline recommendations are unchanged."


def _error(recommendation_id: str, dataset_ref: str, column_ref: str | None, message: str) -> StoredExplanation:
    return StoredExplanation(
        recommendation_id=recommendation_id,
        dataset_ref=dataset_ref,
        column_ref=column_ref,
        status="error",
        error=message,
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def payload_for(recommendation: object) -> dict[str, Any]:
    """Build the consent payload from a recommendation."""
    from universal_dq.recommendations import Recommendation

    assert isinstance(recommendation, Recommendation)
    return explanation_payload(recommendation)
