"""Small, SDK-independent adapters for the three providers' native REST APIs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """A safe-to-display error; never includes request headers or API keys."""


@dataclass
class Completion:
    text: str
    model: str
    response_id: str | None
    usage: dict


def post_json(url, headers, body, timeout):
    request = Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise ProviderError(f"Provider returned HTTP {exc.code}. Check credentials, model access, quota and request limits.") from None
    except (URLError, TimeoutError, OSError):
        raise ProviderError("Provider connection failed or timed out. No automatic retry was made.") from None
    except (ValueError, UnicodeError):
        raise ProviderError("Provider returned an invalid JSON response.") from None


def complete(provider, model, api_key, prompt, *, max_tokens=2048, transport=None):
    """One request, no retries or silent provider/model substitution.

    Temperature is deliberately omitted for models that do not support it.
    Only public topic evidence is sent; no tools, files or browsing are enabled.
    """
    if provider not in {"openai", "anthropic", "gemini"}:
        raise ValueError("Unsupported provider")
    if not api_key.strip() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}", model):
        raise ValueError("Enter an API key and a valid model identifier.")
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        url = "https://api.openai.com/v1/responses"
        headers["Authorization"] = "Bearer " + api_key.strip()
        body = {"model": model, "input": prompt, "max_output_tokens": max_tokens, "store": False}
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers.update({"x-api-key": api_key.strip(), "anthropic-version": "2023-06-01"})
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers["x-goog-api-key"] = api_key.strip()
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "responseMimeType": "application/json"}}
    response = (transport or post_json)(url, headers, body, 60)
    try:
        if provider == "openai":
            if response.get("status") != "completed":
                raise ProviderError("OpenAI did not return a completed response; retry explicitly if appropriate.")
            parts = [part for item in response.get("output", []) if item.get("type") == "message"
                     for part in item.get("content", [])]
            if any(p.get("type") == "refusal" for p in parts):
                raise ProviderError("Provider declined this labeling request.")
            text = "".join(p["text"] for p in parts if p.get("type") == "output_text")
            resolved, usage, rid = response.get("model", model), response.get("usage", {}), response.get("id")
        elif provider == "anthropic":
            if response.get("stop_reason") != "end_turn":
                raise ProviderError("Claude response was truncated or did not finish normally.")
            text = "".join(p["text"] for p in response["content"] if p.get("type") == "text")
            resolved, usage, rid = response.get("model", model), response.get("usage", {}), response.get("id")
        else:
            candidate = response.get("candidates", [{}])[0]
            if candidate.get("finishReason") != "STOP":
                raise ProviderError("Gemini returned no complete candidate (blocked, truncated or unavailable).")
            text = "".join(p.get("text", "") for p in candidate["content"]["parts"] if not p.get("thought"))
            resolved, usage, rid = response.get("modelVersion", model), response.get("usageMetadata", {}), response.get("responseId")
        if not text.strip():
            raise ProviderError("Provider returned no label text.")
        return Completion(text, resolved, rid, usage)
    except (KeyError, IndexError, TypeError):
        raise ProviderError("Provider response did not match the expected response format.") from None
