"""Utilities for running OpenAI Batch jobs from HypotheSAEs."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from .llm_api import get_client

# Default locations and limits can be overridden with env vars.
_DEFAULT_BATCH_DIR = Path(__file__).parent.parent / "batch_cache"
BATCH_CACHE_DIR = Path(os.getenv("HYPOTHESAES_BATCH_DIR", _DEFAULT_BATCH_DIR))
BATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MAX_REQUESTS_PER_JOB = int(os.getenv("HYPOTHESAES_BATCH_MAX_REQUESTS", "50000"))
DEFAULT_POLL_INTERVAL = float(os.getenv("HYPOTHESAES_BATCH_POLL_INTERVAL", "5"))
def _get_completion_window() -> str:
    # As of openai==1.105.0, only '24h' is allowed
    val = os.getenv("HYPOTHESAES_BATCH_COMPLETION_WINDOW", "24h").strip()
    return "24h" if val not in {"24h"} else val


@dataclass
class BatchRequest:
    """Representation of a single request within an OpenAI Batch job."""

    custom_id: str
    method: Literal["POST"] = "POST"
    url: str = "/v1/chat/completions"
    body: Dict[str, Any] = None

    def to_jsonl(self) -> str:
        return json.dumps({
            "custom_id": self.custom_id,
            "method": self.method,
            "url": self.url,
            "body": self.body or {},
        })


@dataclass
class BatchResult:
    """Parsed result from one or more finished batch jobs."""

    responses: Dict[str, Dict[str, Any]]
    errors: Dict[str, Dict[str, Any]]


def choose_backend(preference: str, job_size: int, threshold: int) -> str:
    """Resolve backend choice based on preference and workload size."""

    normalized = preference.lower()
    if normalized not in {"live", "batch", "auto"}:
        raise ValueError("backend must be 'live', 'batch', or 'auto'")
    if normalized != "auto":
        return normalized
    return "batch" if job_size >= threshold else "live"


class OpenAIBatchExecutor:
    """Helper for submitting and polling OpenAI Batch jobs."""

    def __init__(
        self,
        *,
        endpoint: str,
        task_name: str,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_requests_per_job: int = DEFAULT_MAX_REQUESTS_PER_JOB,
        output_dir: Optional[Path] = None,
        completion_window: str = _get_completion_window(),
    ) -> None:
        self.client = get_client()
        self.endpoint = endpoint
        self.task_name = task_name
        self.poll_interval = poll_interval
        self.max_requests_per_job = max_requests_per_job
        self.output_dir = output_dir or BATCH_CACHE_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.completion_window = completion_window

    def execute(
        self,
        requests: List[BatchRequest],
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> BatchResult:
        if not requests:
            return BatchResult(responses={}, errors={})

        aggregated_responses: Dict[str, Dict[str, Any]] = {}
        aggregated_errors: Dict[str, Dict[str, Any]] = {}
        for chunk_idx, chunk in enumerate(_chunk_iterable(requests, self.max_requests_per_job)):
            chunk_list = list(chunk)
            chunk_result = self._execute_chunk(
                chunk_list,
                metadata={**(metadata or {}), "chunk": str(chunk_idx)},
            )
            aggregated_responses.update(chunk_result.responses)
            aggregated_errors.update(chunk_result.errors)

        return BatchResult(responses=aggregated_responses, errors=aggregated_errors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_chunk(
        self,
        requests: List[BatchRequest],
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> BatchResult:
        temp_path = self._write_jsonl(requests)
        try:
            with open(temp_path, "rb") as f:
                input_file = self.client.files.create(file=f, purpose="batch")
        finally:
            os.remove(temp_path)

        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint=self.endpoint,
            completion_window=self.completion_window,
            metadata={"task": self.task_name, **(metadata or {})},
        )

        batch_dir = self.output_dir / f"{self.task_name}-{batch.id}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(batch_dir / "request_summary.json", {
            "batch_id": batch.id,
            "request_count": len(requests),
            "endpoint": self.endpoint,
            "metadata": metadata,
        })

        final_state = self._poll_batch(batch.id)
        self._write_json(batch_dir / "response_summary.json", final_state)

        if final_state.get("status") != "completed":
            message = final_state.get("last_error", {}).get("message", "Unknown batch failure")
            raise RuntimeError(f"Batch {batch.id} failed with status {final_state.get('status')}: {message}")

        output_file_id = final_state.get("output_file_id")
        if output_file_id is None:
            raise RuntimeError(f"Batch {batch.id} completed without an output file")

        output_bytes = self.client.files.content(output_file_id)
        output_text = _coerce_bytes_to_text(output_bytes)

        responses: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, Dict[str, Any]] = {}
        for line in output_text.strip().splitlines():
            if not line:
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id")
            if not custom_id:
                continue
            if "response" in record:
                responses[custom_id] = record["response"].get("body", {})
            elif "error" in record:
                errors[custom_id] = record["error"]

        self._write_json(batch_dir / "parsed_output.json", {
            "responses": list(responses.keys()),
            "errors": errors,
        })

        return BatchResult(responses=responses, errors=errors)

    def _poll_batch(self, batch_id: str) -> Dict[str, Any]:
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = getattr(batch, "status", None)
            if status not in {"queued", "validating", "in_progress", "finalizing", "cancelling"}:
                return {
                    "status": status,
                    "output_file_id": getattr(batch, "output_file_id", None),
                    "last_error": getattr(batch, "last_error", None),
                }
            time.sleep(self.poll_interval)

    @staticmethod
    def _write_jsonl(requests: List[BatchRequest]) -> str:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            for request in requests:
                tmp.write(request.to_jsonl())
                tmp.write("\n")
            return tmp.name

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _chunk_iterable(iterable: Iterable[BatchRequest], size: int) -> Iterable[List[BatchRequest]]:
    chunk: List[BatchRequest] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _coerce_bytes_to_text(response: Any) -> str:
    """Normalize response objects returned by client.files.content()."""

    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return str(content)
    if isinstance(response, bytes):
        return response.decode("utf-8")
    return str(response)


def make_custom_id(prefix: str) -> str:
    """Return a collision-resistant custom_id for batch requests."""

    return f"{prefix}-{uuid.uuid4()}"
