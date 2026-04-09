from __future__ import annotations

from typing import Any, Literal

import ollama
import streamlit as st

try:
    from . import ollama_core
except ImportError:
    import ollama_core


def _run_pull_stream_api(progress: Any, status_box: Any, model_name: str) -> tuple[bool, str]:
    last_line = ""
    for chunk in ollama_core.pull_stream_api(model_name):
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
                text=f"{pct}% ({ollama_core.fmt_bytes(completed)} / {ollama_core.fmt_bytes(total)})",
            )
        elif line:
            progress.progress(0.0, text=line[:80])
    progress.progress(1.0, text="Complete")
    return True, last_line or "Done"


def run_pull_with_progress(model_name: str) -> tuple[bool, str]:
    model_name = (model_name or "").strip()
    progress = st.progress(0.0, text="Starting…")
    status_box = st.empty()
    last_line = ""
    cli_note = ""

    try:
        gen = ollama_core.iter_pull_cli_lines(model_name)
        while True:
            try:
                line = next(gen)
            except StopIteration as stop:
                code = int(stop.value or 0)
                break
            last_line = line
            status_box.caption(line)
            progress.progress(0.0, text=line[:120])
        if code == 0:
            progress.progress(1.0, text="Complete")
            return True, last_line or "Done"
        cli_note = last_line or f"CLI exited with code {code}"
    except FileNotFoundError:
        cli_note = "`ollama` not found on PATH."
    except Exception as e:
        cli_note = str(e)

    status_box.caption("CLI pull did not finish. Trying Ollama API (same as a working `ollama serve`)…")
    try:
        _ok, msg = _run_pull_stream_api(progress, status_box, model_name)
        return True, msg
    except Exception as e:
        progress.progress(0.0, text="Failed")
        hint = ""
        if "ollama.com" in (cli_note or "").lower():
            hint = " Your `ollama` command may be a stub; install the real binary from https://ollama.com/download or use a working daemon."
        return False, f"{cli_note} | API: {e}.{hint}"


def render_model_library_panel(client: ollama.Client, models: list[Any], list_error: str | None) -> None:
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
            ok, msg = run_pull_with_progress(name)
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

    df = ollama_core.models_to_dataframe(models)
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


def get_chat_think_param(
    client: ollama.Client, model: str
) -> bool | Literal["low", "medium", "high"] | None:
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
    legacy_messages = st.session_state.get("messages")

    if "chats" not in st.session_state:
        st.session_state.chats: dict[int, dict] = {}
    if "next_chat_id" not in st.session_state:
        st.session_state.next_chat_id = 1
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id: int | None = None
    if "model_dock_open" not in st.session_state:
        st.session_state.model_dock_open = False

    def _create_chat(name: str | None = None) -> int:
        chat_id = st.session_state.next_chat_id
        st.session_state.next_chat_id += 1
        if name is None:
            name = f"Chat {chat_id}"
        st.session_state.chats[chat_id] = {"id": chat_id, "name": name, "messages": []}
        st.session_state.active_chat_id = chat_id
        return chat_id

    if not st.session_state.chats:
        first_id = _create_chat("Chat 1")
        if isinstance(legacy_messages, list) and legacy_messages:
            st.session_state.chats[first_id]["messages"].extend(legacy_messages)
        if "messages" in st.session_state:
            del st.session_state["messages"]


def get_active_chat() -> dict:
    if "chats" not in st.session_state or not st.session_state.chats:
        init_session()

    active_id = st.session_state.get("active_chat_id")
    if active_id is None or active_id not in st.session_state.chats:
        active_id = sorted(st.session_state.chats.keys())[0]
        st.session_state.active_chat_id = active_id
    return st.session_state.chats[active_id]


def main() -> None:
    st.set_page_config(page_title="Ollama UI", page_icon="🦙", layout="wide")
    init_session()

    models_raw, list_error = ollama_core.list_models_raw()
    model_names = sorted([m.model for m in models_raw if m.model])
    client = ollama_core.client()

    st.title("Ollama")
    st.caption("Local models · service control · streaming chat")

    model = ""
    with st.sidebar:
        st.subheader("Ollama service")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start", width="stretch", help="Start the Ollama background service"):
                code, out, err = ollama_core.run_ollama_service("start")
                if code == 0:
                    st.success("Started.")
                else:
                    st.error(err or out or f"Exit {code}")
                st.rerun()
        with col_b:
            if st.button("Stop", width="stretch", help="Stop the Ollama background service"):
                code, out, err = ollama_core.run_ollama_service("stop")
                if code == 0:
                    st.success("Stopped.")
                else:
                    st.error(err or out or f"Exit {code}")
                st.rerun()

        active = ollama_core.ollama_service_active()
        if active is True:
            st.success("Ollama service: **running**")
        elif active is False:
            st.warning("Ollama service: **stopped**")
        else:
            st.info("Ollama service: could not detect state (permissions or install layout?)")

        st.divider()
        st.subheader("Downloaded models")
        if list_error:
            st.caption(f"Cannot reach Ollama at `{ollama_core.OLLAMA_HOST}`. Is the daemon running?")
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
                    code, out, err = ollama_core.ollama_stop(previous_model)
                    if code != 0:
                        st.toast(f"Failed to stop {previous_model}: {err or out or f'Exit {code}'}")
                    else:
                        st.toast(f"Stopped {previous_model}")
                st.session_state.active_model_name = model
            st.caption(f"Using **{model}**")
        elif list_error is None:
            st.caption(
                "No models yet — open **Models** on the right edge of the page, or run `ollama pull` in a terminal."
            )

        st.divider()
        st.subheader("Chats")

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
                new_id = st.session_state.next_chat_id
                st.session_state.next_chat_id += 1
                name = f"Chat {new_id}"
                st.session_state.chats[new_id] = {"id": new_id, "name": name, "messages": []}
                st.session_state.active_chat_id = new_id
                st.rerun()

        with col_delete:
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
                    for kind, piece in ollama_core.chat_stream(
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
                        ollama_core.ollama_stop(model)

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

