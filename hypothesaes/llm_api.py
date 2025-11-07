"""LLM API utilities for HypotheSAEs."""

import os
import time
import openai
import logging
import pdb

_CLIENT_OPENAI = None  # Module-level cache for the OpenAI client

"""
These model IDs point to the latest versions of the models as of 2025-05-04.
We point to a specific version for reproducibility, but feel free to update them as necessary.
Note that o-series models (o1, o1-mini, o3-mini) are also supported by get_completion().
We don't point these models to a specific version, so passing in these model names will use the latest version.

2025-05-04:
- Removed gpt-4 (deprecated by gpt-4o, will be removed from API soon)
- Added gpt-4.1 models (not used by HypotheSAEs paper, but potentially of interest)

2025-03-12:
- First version of this file: supports gpt-4o, gpt-4o-mini, gpt-4
"""
model_abbrev_to_id = {
    'gpt4o': 'gpt-4o-2024-11-20',
    'gpt-4o': 'gpt-4o-2024-11-20',
    'gpt4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',

    "gpt4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt5": "gpt-5",
    "gpt-5": "gpt-5",
    "gpt5-mini": "gpt-5-mini",
    "gpt-5-mini": "gpt-5-mini",
}

DEFAULT_MODEL = "gpt-5-mini"


def resolve_model_name(model: str) -> str:
    """Return the fully qualified model id for an alias."""

    return model_abbrev_to_id.get(model, model)

def get_client():
    """Get the OpenAI client, initializing it if necessary and caching it."""
    global _CLIENT_OPENAI
    if _CLIENT_OPENAI is not None:
        return _CLIENT_OPENAI

    _configure_openai_logging()

    api_key = os.environ.get('OPENAI_KEY_SAE')
    if api_key is None or '...' in api_key:
        raise ValueError("Please set the OPENAI_KEY_SAE environment variable before using functions which require the OpenAI API.")

    _CLIENT_OPENAI = openai.OpenAI(api_key=api_key)
    return _CLIENT_OPENAI

def _configure_openai_logging() -> None:
    """Suppress verbose HTTP logs from the OpenAI SDK unless explicitly enabled.

    By default we reduce noise so tqdm progress bars render cleanly.
    Set OPENAI_LOG=info|debug or HYPOTHESAES_SUPPRESS_HTTP_LOGS=0 to disable suppression.
    """
    suppress = os.environ.get("HYPOTHESAES_SUPPRESS_HTTP_LOGS", "1").lower() in {"1", "true", "yes"}
    explicit_verbose = os.environ.get("OPENAI_LOG", "").lower() in {"info", "debug"}
    if not suppress or explicit_verbose:
        return

    for name in ("openai", "openai._base_client", "httpx", "httpcore"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

def get_completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 300.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    **kwargs
) -> str:
    """
    Get completion from OpenAI API with retry logic and timeout.
    
    Args:
        prompt: The prompt to send
        model: Model to use
        max_retries: Maximum number of retries on rate limit
        backoff_factor: Factor to multiply backoff time by after each retry
        timeout: Timeout for the request
        **kwargs: Additional arguments to pass to the OpenAI API; max_tokens, temperature, etc.
    Returns:
        Generated completion text
    
    Raises:
        Exception: If all retries fail
    """
    client = get_client()
    model_id = resolve_model_name(model)
    # Always use GPT-5 family via Responses API
    if not str(model_id).startswith("gpt-5"):
        model_id = resolve_model_name("gpt-5-mini")
    
    for attempt in range(max_retries):
        try:
            # Always use the Responses API
            resp_kwargs = dict(kwargs)
            # Map legacy kwarg names
            max_comp = resp_kwargs.pop("max_completion_tokens", None)
            max_tok = resp_kwargs.pop("max_tokens", None)
            if max_comp is not None:
                resp_kwargs["max_output_tokens"] = max_comp
            elif max_tok is not None:
                resp_kwargs["max_output_tokens"] = max_tok
            # Enforce a generous default for GPT-5 if not provided
            if "max_output_tokens" not in resp_kwargs:
                resp_kwargs["max_output_tokens"] = 5000
            effort = resp_kwargs.pop("reasoning_effort", None)
            resp_kwargs["reasoning"] = {"effort": effort or 'low'}
            # Prefer concise text outputs to fit within token budget
            if "text" not in resp_kwargs:
                resp_kwargs["text"] = {"verbosity": "low"}
            response = client.responses.create(
                model=model_id,
                input=prompt,
                timeout=timeout,
                **resp_kwargs,
            )
            # Raise if the response is not completed (e.g., truncated/incomplete)
            status = getattr(response, "status", None)
            if status and status != "completed":
                details = getattr(response, "incomplete_details", None)
                pdb.set_trace()
                raise RuntimeError(f"Responses API returned status={status}; details={details}")
            text = _extract_output_text(response)
            return text
            
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            wait_time = timeout * (backoff_factor ** attempt)
            if attempt > 0:
                print(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)


def _extract_output_text(response_obj) -> str:
    """Best-effort extraction of text from Responses API objects.

    Prefers the SDK's convenience attribute 'output_text'. Falls back to
    inspecting the dumped dict for 'output_text' or composing text pieces from
    'output[*].content[*].text'.
    """
    # SDK convenience
    text = getattr(response_obj, "output_text", None)
    if isinstance(text, str) and text:
        return text

    # Try model_dump() to dict
    dump = None
    for attr in ("model_dump", "to_dict"):
        fn = getattr(response_obj, attr, None)
        if callable(fn):
            try:
                dump = fn()
                break
            except Exception:
                pass
    if isinstance(dump, dict):
        if isinstance(dump.get("output_text"), str) and dump["output_text"]:
            return dump["output_text"]
        out = dump.get("output") or []
        pieces = []
        for item in out:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for part in content:
                    txt = part.get("text") if isinstance(part, dict) else None
                    if isinstance(txt, str):
                        pieces.append(txt)
        if pieces:
            return "".join(pieces)
    # Fallback: return empty string to signal "no text available"
    return ""
