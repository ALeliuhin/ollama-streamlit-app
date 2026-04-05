"""Streamlit UI for Ollama: service control, model list, and streaming chat."""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Iterator, Literal

import ollama
import pandas as pd
import streamlit as st

OLLAMA_HOST = "http://127.0.0.1:11434"

def ollama_stop(model: str) -> tuple[int, str, str]:
    """Stop/unload a model via `ollama stop <model>`."""
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


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def run_ollama_service(action: str) -> tuple[int, str, str]:
    """Start/stop/query the Ollama OS service. action: start|stop|status|is-active."""
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
    """True if active, False if inactive, None if status could not be determined."""
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
    """Return (models from Ollama list(), None) or ([], error_message)."""
    try:
        client = _ollama_client()
        response = client.list()
        return list(response.models), None
    except Exception as e:
        return [], str(e)


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _models_to_dataframe(models: list[Any]) -> pd.DataFrame:
    rows = []
    for m in models:
        d = getattr(m, "details", None)
        rows.append(
            {
                "name": m.model or "—",
                "size": _fmt_bytes(m.size),
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


def _run_pull_stream_api(client: ollama.Client, model_name: str, progress: Any, status_box: Any) -> tuple[bool, str]:
    """Stream pull via Ollama HTTP API (works when the real daemon is up)."""
    last_line = ""
    stream = client.pull(model_name.strip(), stream=True)
    for chunk in stream:
        line = getattr(chunk, "status", None) or ""
        if line:
            last_line = line
            status_box.caption(line)
        total = getattr(chunk, "total", None)
        completed = getattr(chunk, "completed", None)
        if total and total > 0 and completed is not None:
            frac = min(float(completed) / float(total), 1.0)
            pct = int(100 * frac)
            progress.progress(
                frac,
                text=f"{pct}% ({_fmt_bytes(completed)} / {_fmt_bytes(total)})",
            )
        elif line:
            progress.progress(0.0, text=line[:80])
    progress.progress(1.0, text="Complete")
    return True, last_line or "Done"


def run_pull_with_progress(client: ollama.Client, model_name: str) -> tuple[bool, str]:
    """Prefer `ollama pull` CLI; if that fails (e.g. distro stub prints ollama.com), use HTTP API."""
    model_name = (model_name or "").strip()
    progress = st.progress(0.0, text="Starting…")
    status_box = st.empty()
    last_line = ""
    cli_note = ""

    try:
        proc = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is None:
            cli_note = "Could not capture `ollama pull` output."
        else:
            for raw in proc.stdout:
                line = raw.strip()
                if line:
                    last_line = line
                    status_box.caption(line)
                    progress.progress(0.0, text=line[:120])
            code = proc.wait()
            if code == 0:
                progress.progress(1.0, text="Complete")
                return True, last_line or "Done"
            cli_note = last_line or f"CLI exited with code {code}"
    except FileNotFoundError:
        cli_note = "`ollama` not found on PATH."
    except Exception as e:
        cli_note = str(e)

    # Many Linux installs ship a stub `ollama` that only prints https://ollama.com/download — use API if daemon works.
    status_box.caption("CLI pull did not finish. Trying Ollama API (same as a working `ollama serve`)…")
    try:
        _ok, msg = _run_pull_stream_api(client, model_name, progress, status_box)
        return True, msg
    except Exception as e:
        progress.progress(0.0, text="Failed")
        hint = ""
        if "ollama.com" in (cli_note or "").lower():
            hint = " Your `ollama` command may be a stub; install the real binary from https://ollama.com/download or use a working daemon."
        return False, f"{cli_note} | API: {e}.{hint}"


def render_model_library_panel(client: ollama.Client, models: list[Any], list_error: str | None) -> None:
    """Right-docked panel: pull models, progress, and property table."""
    st.subheader("Model library")
    st.caption("Pull new models and inspect installed weights.")

    if list_error:
        st.warning(f"Cannot list models: {list_error}")

    with st.form("pull_model_form", clear_on_submit=False):
        pull_name = st.text_input(
            "Model name",
            placeholder="e.g. llama3.2 or qwen2.5:7b",
            help="Same name you would pass to `ollama pull` on the CLI.",
        )
        submitted = st.form_submit_button("Pull", width="stretch")

    if submitted:
        name = (pull_name or "").strip()
        if not name:
            st.warning("Enter a model name to pull.")
        else:
            ok, msg = run_pull_with_progress(client, name)
            if ok:
                st.success(f"Finished pulling **{name}**.")
                st.rerun()
            else:
                st.error(msg)

    st.divider()
    st.markdown("**Installed models**")

    if list_error and not models:
        st.caption("Connect Ollama to see installed models here.")
        return

    if not models:
        st.caption("No models yet — pull one above.")
        return

    df = _models_to_dataframe(models)
    st.dataframe(
        df.drop(columns=["size_bytes"], errors="ignore"),
        width="stretch",
        hide_index=True,
        height=min(280, 36 + len(models) * 36),
    )

    names = [m.model for m in models if m.model]
    if not names:
        return

    detail_name = st.selectbox("Inspect details", options=sorted(names), key="inspect_model_select")
    if st.button("Load details", key="load_show_details"):
        try:
            st.session_state["ollama_show_payload"] = client.show(detail_name)
            st.session_state["ollama_show_for"] = detail_name
            st.session_state.pop("ollama_show_err", None)
        except Exception as e:
            st.session_state["ollama_show_payload"] = None
            st.session_state["ollama_show_err"] = str(e)

    err = st.session_state.get("ollama_show_err")
    if err:
        st.error(err)

    info = st.session_state.get("ollama_show_payload")
    cached_for = st.session_state.get("ollama_show_for")
    if info is None or cached_for != detail_name:
        st.caption("Pick a model and click **Load details** for full `ollama show` metadata.")
        return

    st.markdown(f"**{detail_name}**")

    det = getattr(info, "details", None)
    if det:
        if hasattr(det, "model_dump"):
            d = det.model_dump()
        else:
            d = dict(det) if det else {}
        st.json({k: v for k, v in d.items() if v not in (None, "", [])})

    caps = getattr(info, "capabilities", None)
    if caps:
        st.caption("Capabilities: " + ", ".join(caps))

    mi = getattr(info, "modelinfo", None)
    if mi:
        with st.expander("Model info (GGUF metadata)", expanded=False):
            st.json(mi if isinstance(mi, dict) else dict(mi))

    modelfile = getattr(info, "modelfile", None)
    if modelfile:
        with st.expander("Modelfile", expanded=False):
            st.code(modelfile[:8000] + ("…" if len(modelfile) > 8000 else ""), language="dockerfile")


def chat_stream(
    model: str,
    messages: list[dict],
    *,
    keep_alive: int | str | None = None,
    think: bool | Literal["low", "medium", "high"] | None = None,
) -> Iterator[tuple[Literal["thinking", "content"], str]]:
    """Stream chat tokens.

    Yields ``("thinking", piece)`` for reasoning traces and ``("content", piece)`` for the reply.
    Pass ``think`` only for models that advertise the ``thinking`` capability (see
    ``get_chat_think_param``); with ``think`` set, thinking-capable models emit ``message.thinking``
    chunks; otherwise only ``content`` pieces are yielded.
    """
    client = _ollama_client()
    kwargs: dict[str, Any] = {}
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    if think is not None:
        kwargs["think"] = think
    stream = client.chat(model=model, messages=messages, stream=True, **kwargs)
    for chunk in stream:
        msg = chunk.message
        if not msg:
            continue
        t = getattr(msg, "thinking", None)
        if t:
            yield "thinking", t
        c = getattr(msg, "content", None)
        if c:
            yield "content", c


def get_chat_think_param(
    client: ollama.Client, model: str
) -> bool | Literal["low", "medium", "high"] | None:
    """Value for Ollama ``think`` on chat requests, or ``None`` to omit (no thinking capability).

    Uses ``/api/show`` ``capabilities``; caches per model name in session state.
    GPT-OSS expects a level instead of a boolean, so we pass ``\"medium\"`` when the model name
    matches that family.
    """
    name = (model or "").strip()
    if not name:
        return None
    cache_key = f"ollama_model_caps::{name}"
    if cache_key not in st.session_state:
        try:
            info = client.show(name)
            caps = getattr(info, "capabilities", None) or []
            st.session_state[cache_key] = tuple(caps)
        except Exception:
            st.session_state[cache_key] = ()
    caps: tuple[Any, ...] = st.session_state[cache_key]
    if "thinking" not in caps:
        return None
    if "gpt-oss" in name.lower():
        return "medium"
    return True


def _format_assistant_stream_markdown(
    thinking: str, content: str, *, include_thinking: bool = True
) -> str:
    parts: list[str] = []
    if include_thinking and thinking:
        parts.append("**Thinking**\n\n")
        parts.append(thinking)
    if content:
        if include_thinking and thinking:
            parts.append("\n\n---\n\n")
        parts.append(content)
    return "".join(parts)


def init_session() -> None:
    # Backwards compatibility: migrate legacy single-chat messages, if any.
    legacy_messages = st.session_state.get("messages")

    if "chats" not in st.session_state:
        st.session_state.chats: dict[int, dict] = {}
    if "next_chat_id" not in st.session_state:
        st.session_state.next_chat_id = 1
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id: int | None = None
    if "model_dock_open" not in st.session_state:
        st.session_state.model_dock_open = False

    # Helper to create an empty chat.
    def _create_chat(name: str | None = None) -> int:
        chat_id = st.session_state.next_chat_id
        st.session_state.next_chat_id += 1
        if name is None:
            name = f"Chat {chat_id}"
        st.session_state.chats[chat_id] = {"id": chat_id, "name": name, "messages": []}
        st.session_state.active_chat_id = chat_id
        return chat_id

    # If there are no chats yet, create the first one.
    if not st.session_state.chats:
        first_id = _create_chat("Chat 1")
        # Migrate old single-chat history if present.
        if isinstance(legacy_messages, list) and legacy_messages:
            st.session_state.chats[first_id]["messages"].extend(legacy_messages)
        # Remove legacy key to avoid confusion.
        if "messages" in st.session_state:
            del st.session_state["messages"]


def get_active_chat() -> dict:
    """Return the currently active chat dict, creating one if needed."""
    if "chats" not in st.session_state or not st.session_state.chats:
        init_session()

    active_id = st.session_state.get("active_chat_id")
    if active_id is None or active_id not in st.session_state.chats:
        # Pick an arbitrary existing chat as active.
        active_id = sorted(st.session_state.chats.keys())[0]
        st.session_state.active_chat_id = active_id
    return st.session_state.chats[active_id]


def main() -> None:
    st.set_page_config(page_title="Ollama UI", page_icon="🦙", layout="wide")
    init_session()

    models_raw, list_error = list_models_raw()
    model_names = sorted([m.model for m in models_raw if m.model])
    client = _ollama_client()

    st.title("Ollama")
    st.caption("Local models · service control · streaming chat")

    model = ""
    with st.sidebar:
        st.subheader("Ollama service")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start", width="stretch", help="Start the Ollama background service"):
                code, out, err = run_ollama_service("start")
                if code == 0:
                    st.success("Started.")
                else:
                    st.error(err or out or f"Exit {code}")
                st.rerun()
        with col_b:
            if st.button("Stop", width="stretch", help="Stop the Ollama background service"):
                code, out, err = run_ollama_service("stop")
                if code == 0:
                    st.success("Stopped.")
                else:
                    st.error(err or out or f"Exit {code}")
                st.rerun()

        active = ollama_service_active()
        if active is True:
            st.success("Ollama service: **running**")
        elif active is False:
            st.warning("Ollama service: **stopped**")
        else:
            st.info("Ollama service: could not detect state (permissions or install layout?)")

        st.divider()
        st.subheader("Downloaded models")
        if list_error:
            st.caption(f"Cannot reach Ollama at `{OLLAMA_HOST}`. Is the daemon running?")
            st.caption(list_error)

        if model_names:
            previous_model = st.session_state.get("active_model_name", "")
            model = st.selectbox(
                "Chat model",
                options=model_names,
                index=0,
                key="ollama_chat_model",
            )
            if model != previous_model:
                if previous_model:
                    code, out, err = ollama_stop(previous_model)
                    if code != 0:
                        st.toast(f"Failed to stop {previous_model}: {err or out or f'Exit {code}'}")
                    else:
                        st.toast(f"Stopped {previous_model}")
                st.session_state.active_model_name = model
            st.caption(f"Using **{model}**")
        elif list_error is None:
            st.caption("No models yet — open **Models** on the right edge of the page, or run `ollama pull` in a terminal.")

        st.divider()
        st.subheader("Chats")

        # Chat selection and management
        chats = st.session_state.chats
        chat_ids = sorted(chats.keys())

        if chat_ids:
            current_active = st.session_state.get("active_chat_id", chat_ids[0])
            try:
                current_index = chat_ids.index(current_active)
            except ValueError:
                current_index = 0
                st.session_state.active_chat_id = chat_ids[0]

            selected_id = st.radio(
                "Select chat",
                options=chat_ids,
                index=current_index,
                key="chat_selector",
                format_func=lambda cid: chats[cid]["name"],
            )
            if selected_id != st.session_state.active_chat_id:
                st.session_state.active_chat_id = selected_id
                st.rerun()

        col_new, col_delete = st.columns(2)
        with col_new:
            if st.button("New chat", width="stretch"):
                # Create a new, empty chat.
                new_id = st.session_state.next_chat_id
                st.session_state.next_chat_id += 1
                name = f"Chat {new_id}"
                st.session_state.chats[new_id] = {"id": new_id, "name": name, "messages": []}
                st.session_state.active_chat_id = new_id
                st.rerun()

        with col_delete:
            # Only allow deleting if more than one chat exists to avoid ending up with none.
            disabled = len(st.session_state.chats) <= 1
            if st.button(
                "Delete chat",
                width="stretch",
                disabled=disabled,
                help="Delete the current chat. At least one chat must remain.",
            ):
                active_id = st.session_state.get("active_chat_id")
                if active_id in st.session_state.chats and len(st.session_state.chats) > 1:
                    del st.session_state.chats[active_id]
                    remaining_ids = sorted(st.session_state.chats.keys())
                    st.session_state.active_chat_id = remaining_ids[0]
                st.rerun()

        st.divider()
        st.subheader("Model runtime")
        unload_after_response = st.toggle(
            "Unload after each response",
            value=bool(st.session_state.get("unload_after_response", False)),
            help=(
                "If enabled, the model will be unloaded after it finishes responding. "
                "This reduces parallel models in `ollama ps`, but may increase latency for the next message."
            ),
        )
        st.session_state.unload_after_response = unload_after_response

    if model_names:
        model = st.session_state.get("ollama_chat_model", model_names[0])

    dock_open = st.session_state.get("model_dock_open", False)
    if dock_open:
        col_chat, col_models = st.columns([2.05, 1], gap="large")
        col_edge = None
    else:
        col_chat, col_edge = st.columns([12, 1], gap="small")
        col_models = None

    with col_chat:
        if list_error:
            st.warning(f"Model list unavailable ({list_error}). Start Ollama from the sidebar.")

        active_chat = get_active_chat()
        messages = active_chat["messages"]

        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if not model_names:
            st.info(
                "Start the Ollama service, then download a model from the **Models** panel "
                "(click **Models** on the right edge of the window). You can pull before any model exists."
            )

        if model_names and (prompt := st.chat_input("Message")):
            messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                thinking_buf = ""
                content_buf = ""
                try:
                    st.session_state.active_generation_model = model
                    keep_alive = 0 if st.session_state.get("unload_after_response") else None
                    think_param = get_chat_think_param(client, model)
                    show_thinking = think_param is not None
                    for kind, piece in chat_stream(
                        model, messages, keep_alive=keep_alive, think=think_param
                    ):
                        if kind == "thinking":
                            thinking_buf += piece
                        else:
                            content_buf += piece
                        placeholder.markdown(
                            _format_assistant_stream_markdown(
                                thinking_buf, content_buf, include_thinking=show_thinking
                            )
                            + "▌"
                        )
                    full = _format_assistant_stream_markdown(
                        thinking_buf, content_buf, include_thinking=show_thinking
                    )
                    placeholder.markdown(full)
                except Exception as e:
                    placeholder.error(str(e))
                    messages.pop()
                else:
                    messages.append({"role": "assistant", "content": full})
                    st.rerun()
                finally:
                    st.session_state.active_generation_model = ""
                    if st.session_state.get("unload_after_response"):
                        ollama_stop(model)

    if col_edge is not None:
        with col_edge:
            if st.button(
                "Models",
                key="open_model_dock",
                width="stretch",
                help="Open model library: pull models and inspect metadata.",
            ):
                st.session_state.model_dock_open = True
                st.rerun()

    if col_models is not None:
        with col_models:
            close_l, close_r = st.columns([3, 1])
            with close_r:
                if st.button(
                    "Close",
                    key="close_model_dock",
                    width="stretch",
                    help="Hide the model library panel.",
                ):
                    st.session_state.model_dock_open = False
                    st.rerun()
            with st.container(border=True):
                render_model_library_panel(client, models_raw, list_error)


if __name__ == "__main__":
    import subprocess
    import sys

    import streamlit.runtime as st_runtime

    # `uv run -m src.app`
    if not st_runtime.exists():
        sys.exit(
            subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]])
        )
    main()
