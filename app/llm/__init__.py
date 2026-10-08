"""OpenAI-compatible chat client: Groq now, W&B Inference on Friday (LLM_PROVIDER=groq|wandb)."""

import os
from functools import lru_cache

import weave
from openai import BadRequestError, OpenAI

from app.config import LLM_PROVIDER, LLM_PROVIDERS


def provider_config(provider: str = LLM_PROVIDER) -> dict:
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(LLM_PROVIDERS)}, got {provider!r}")
    cfg = LLM_PROVIDERS[provider]
    key = os.getenv(cfg["key_env"], "")
    if not key:
        raise RuntimeError(f"{cfg['key_env']} is not set (needed for LLM_PROVIDER={provider})")
    return {"provider": provider, "base_url": cfg["base_url"], "api_key": key,
            "model": os.getenv(cfg["model_env"]) or cfg["default_model"],
            "json_mode": cfg.get("json_mode", True), "extra": cfg.get("extra", {})}


@lru_cache(maxsize=4)
def get_client(provider: str = LLM_PROVIDER) -> tuple[OpenAI, str]:
    cfg = provider_config(provider)
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=60, max_retries=2), cfg["model"]


def failed_generation(e: BadRequestError) -> str | None:
    """The rejected text of a json_validate_failed error, if that is what this is."""
    body = e.body if isinstance(e.body, dict) else {}
    err = body.get("error", body) if isinstance(body.get("error", body), dict) else body
    if err.get("code") == "json_validate_failed" and isinstance(err.get("failed_generation"), str):
        return err["failed_generation"]
    return None


@weave.op(name="llm_chat_json")
def chat_json(messages: list[dict], provider: str = LLM_PROVIDER, temperature: float = 0.2) -> str:
    """One chat completion that should return a JSON object; returns the raw text."""
    client, model = get_client(provider)
    cfg = provider_config(provider)
    kwargs = {"model": model, "messages": messages, "temperature": temperature, **cfg["extra"]}
    if not cfg["json_mode"]:
        resp = client.chat.completions.create(**kwargs)
    else:
        try:
            resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except BadRequestError as e:
            failed = failed_generation(e)
            if failed is not None:
                # Groq's JSON mode rejects slightly malformed output (e.g. one stray bracket) with a 400 and
                # returns it as failed_generation. Hand it back: the callers' lenient parser repairs bracket
                # mistakes and their retry loop asks the model again for anything worse.
                return failed
            if "response_format" not in str(e):
                raise
            resp = client.chat.completions.create(**kwargs)  # provider/model without JSON mode
    return resp.choices[0].message.content or ""
