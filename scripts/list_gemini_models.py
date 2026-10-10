"""Print Gemini model ids that support generateContent.

Reads GEMINI_API_KEY from the environment. Prints model ids only.
Does not generate content and does not start the investigator.
"""

from __future__ import annotations

import os
import re
import sys


def _redact(text: str, secret: str) -> str:
    cleaned = text.replace(secret, "<redacted>") if secret else text
    return re.sub(r"AIza[0-9A-Za-z_\-]{8,}|AQ\.[0-9A-Za-z_\-]{8,}", "<redacted>", cleaned)


def main() -> int:
    secret = os.environ.get("GEMINI_API_KEY", "").strip()
    if not secret:
        print(
            "GEMINI_API_KEY is missing or blank in this process. No request was made.",
            file=sys.stderr,
        )
        return 1

    from google import genai

    client = genai.Client(api_key=secret)
    names: list[str] = []
    try:
        for model in client.models.list():
            actions = [action.casefold() for action in (model.supported_actions or [])]
            if "generatecontent" not in actions:
                continue
            resource_name = model.name or ""
            short_name = resource_name.rsplit("/", 1)[-1].strip()
            if short_name:
                names.append(short_name)
    except Exception as exc:
        print(f"Could not list models: {_redact(str(exc), secret)}", file=sys.stderr)
        return 1

    for name in sorted(set(names)):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
