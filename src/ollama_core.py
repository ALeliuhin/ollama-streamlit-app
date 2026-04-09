from __future__ import annotations

import subprocess
import sys
from typing import Any, Iterator, Literal

import ollama
import pandas as pd

OLLAMA_HOST = "http://127.0.0.1:11434"


def ollama_stop(model: str) -> tuple[int, str, str]:
    model = (model or "").strip()
    if not model:
        return 0, "", ""
    result = subprocess.run(
        ["ollama", "stop", model],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def run_ollama_service(action: str) -> tuple[int, str, str]:
    if action not in ("start", "stop", "status", "is-active"):
        raise ValueError("invalid service action")
    if sys.platform == "win32":
        if action in ("start", "stop"):
            cmd = ["net", action, "Ollama"]
        else:
            cmd = ["sc", "query", "Ollama"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    result = subprocess.run(
        ["systemctl", action, "ollama"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def ollama_service_active() -> bool | None:
    if sys.platform == "win32":
        code, out, _err = run_ollama_service("is-active")
        if code != 0:
            return None
        u = out.upper()
        if "RUNNING" in u:
            return True
        if "STOPPED" in u:
            return False
        return None
    code, out, _err = run_ollama_service("is-active")
    if code == 0 and out == "active":
        return True
    if code in (3, 0) and out == "inactive":
        return False
    return None


def list_models_raw() -> tuple[list[Any], str | None]:
    try:
        response = client().list()
        return list(response.models), None
    except Exception as e:
        return [], str(e)


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def models_to_dataframe(models: list[Any]) -> pd.DataFrame:
    rows = []
    for m in models:
        d = getattr(m, "details", None)
        rows.append(
            {
                "name": m.model or "—",
                "size": fmt_bytes(m.size),
                "size_bytes": m.size,
                "modified": m.modified_at.isoformat() if m.modified_at else "—",
                "format": (getattr(d, "format", None) or "—") if d else "—",
                "family": (getattr(d, "family", None) or "—") if d else "—",
                "parameter_size": (getattr(d, "parameter_size", None) or "—") if d else "—",
                "quantization": (getattr(d, "quantization_level", None) or "—") if d else "—",
                "digest": (m.digest[:16] + "…") if m.digest and len(m.digest) > 16 else (m.digest or "—"),
            }
        )
    return pd.DataFrame(rows)


def pull_stream_api(model_name: str) -> Iterator[Any]:
    model_name = (model_name or "").strip()
    stream = client().pull(model_name, stream=True)
    for chunk in stream:
        yield chunk


def iter_pull_cli_lines(model_name: str) -> Iterator[str]:
    model_name = (model_name or "").strip()
    proc = subprocess.Popen(
        ["ollama", "pull", model_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is None:
        return
    for raw in proc.stdout:
        line = raw.strip()
        if line:
            yield line
    return proc.wait()


def chat_stream(
    model: str,
    messages: list[dict],
    *,
    keep_alive: int | str | None = None,
    think: bool | Literal["low", "medium", "high"] | None = None,
) -> Iterator[tuple[Literal["thinking", "content"], str]]:
    c = client()
    kwargs: dict[str, Any] = {}
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    if think is not None:
        kwargs["think"] = think
    stream = c.chat(model=model, messages=messages, stream=True, **kwargs)
    for chunk in stream:
        msg = chunk.message
        if not msg:
            continue
        t = getattr(msg, "thinking", None)
        if t:
            yield "thinking", t
        content = getattr(msg, "content", None)
        if content:
            yield "content", content

