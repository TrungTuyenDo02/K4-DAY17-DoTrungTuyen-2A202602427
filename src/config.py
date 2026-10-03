from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

# Default model per provider when LLM_MODEL is not set.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

# Env var holding the API key / base URL for each provider.
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}
BASE_URL_ENV = {
    "openai": "OPENAI_BASE_URL",
    "custom": "CUSTOM_BASE_URL",
    "gemini": None,
    "anthropic": "ANTHROPIC_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab: paths, compact-memory knobs, models."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    # Bonus knobs (have defaults so tests can build a LabConfig with the 7 core fields).
    profile_confidence_threshold: float = 0.6
    profile_max_interests: int = 8
    live_mode: bool = False


def _env(name: str | None, default: str | None = None) -> str | None:
    if not name:
        return default
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Read `<prefix>_PROVIDER`, `<prefix>_MODEL`, ... with per-provider key lookup."""

    provider = normalize_provider(_env(f"{prefix}_PROVIDER", fallback.provider if fallback else "openai"))
    same_provider = fallback is not None and fallback.provider == provider
    model_name = _env(f"{prefix}_MODEL", fallback.model_name if same_provider else DEFAULT_MODELS[provider])
    temperature = float(_env(f"{prefix}_TEMPERATURE", str(fallback.temperature) if fallback else "0.2"))
    api_key = _env(f"{prefix}_API_KEY", _env(API_KEY_ENV[provider]))
    if provider == "gemini" and not api_key:
        api_key = _env("GOOGLE_API_KEY")
    base_url = _env(f"{prefix}_BASE_URL", _env(BASE_URL_ENV[provider]))
    return ProviderConfig(provider, model_name, temperature, api_key=api_key, base_url=base_url)


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` + environment variables and return a populated LabConfig.

    Env vars:
    - LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE / LLM_API_KEY / LLM_BASE_URL
    - JUDGE_PROVIDER / JUDGE_MODEL ... (defaults to the main model)
    - provider keys: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY,
      OPENROUTER_API_KEY, CUSTOM_API_KEY + CUSTOM_BASE_URL, OLLAMA_BASE_URL
    - COMPACT_THRESHOLD_TOKENS / COMPACT_KEEP_MESSAGES
    - PROFILE_CONFIDENCE_THRESHOLD / PROFILE_MAX_INTERESTS
    - LAB_LIVE=1 to call the real model (offline deterministic mode otherwise)
    - LAB_STATE_DIR to override `state/`
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    state_dir = Path(_env("LAB_STATE_DIR", str(root / "state"))).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(_env("COMPACT_THRESHOLD_TOKENS", "800")),
        compact_keep_messages=int(_env("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
        profile_confidence_threshold=float(_env("PROFILE_CONFIDENCE_THRESHOLD", "0.6")),
        profile_max_interests=int(_env("PROFILE_MAX_INTERESTS", "8")),
        live_mode=_env("LAB_LIVE", "0").lower() in ("1", "true", "yes", "on"),
    )
