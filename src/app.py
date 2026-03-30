"""Streamlit UI for Ollama: service control, model list, and streaming chat."""

from __future__ import annotations

import subprocess
from typing import Iterator

import ollama
import streamlit as st

OLLAMA_HOST = "http://127.0.0.1:11434"


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def run_systemctl(action: str) -> tuple[int, str, str]:
    """Run systemctl start|stop|status ollama. Returns (code, stdout, stderr)."""
    if action not in ("start", "stop", "status", "is-active"):
        raise ValueError("invalid systemctl action")
    result = subprocess.run(
        ["systemctl", action, "ollama"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def ollama_service_active() -> bool | None:
    """True if active, False if inactive, None if status could not be determined."""
    code, out, _err = run_systemctl("is-active")
    if code == 0 and out == "active":
        return True
    if code in (3, 0) and out == "inactive":
        return False
    return None


def list_models() -> list[str]:
    client = _ollama_client()
    response = client.list()
    names = [m.model for m in response.models if m.model]
    return sorted(names)


def chat_stream(model: str, messages: list[dict]) -> Iterator[str]:
    client = _ollama_client()
    stream = client.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        piece = chunk.message.content if chunk.message else None
        if piece:
            yield piece


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(page_title="Ollama UI", page_icon="🦙", layout="wide")
    init_session()

    st.title("Ollama")
    st.caption("Local models · systemd service · streaming chat")

    models: list[str] = []
    list_error: str | None = None
    with st.sidebar:
        st.subheader("Ollama service")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start", use_container_width=True, help="systemctl start ollama"):
                code, out, err = run_systemctl("start")
                if code == 0:
                    st.success("Started.")
                else:
                    st.error(err or out or f"Exit {code}")
                st.rerun()
        with col_b:
            if st.button("Stop", use_container_width=True, help="systemctl stop ollama"):
                code, out, err = run_systemctl("stop")
                if code == 0:
                    st.success("Stopped.")
                else:
                    st.error(err or out or f"Exit {code}")
                st.rerun()

        active = ollama_service_active()
        if active is True:
            st.success("systemd: **active**")
        elif active is False:
            st.warning("systemd: **inactive**")
        else:
            st.info("systemd: could not read `is-active` (permissions?)")

        st.divider()
        st.subheader("Downloaded models")
        try:
            models = list_models()
        except Exception as e:
            list_error = str(e)
            st.caption(f"Cannot reach Ollama at `{OLLAMA_HOST}`. Is the daemon running?")
            st.caption(list_error)
        else:
            list_error = None

        if models:
            for name in models:
                st.text(name)
        elif list_error is None:
            st.caption("No models yet — run `ollama pull <name>` in a terminal.")

    if list_error:
        st.warning(f"Model list unavailable ({list_error}). Start Ollama from the sidebar.")

    model = ""
    if models:
        model = st.selectbox(
            "Chat model",
            options=models,
            index=0,
            key="ollama_chat_model",
        )
        st.caption(f"Using **{model}**")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not models:
        st.info("Start the Ollama service and ensure at least one model is downloaded.")
        return

    if prompt := st.chat_input("Message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            try:
                for piece in chat_stream(model, st.session_state.messages):
                    full += piece
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
            except Exception as e:
                placeholder.error(str(e))
                st.session_state.messages.pop()
                return

        st.session_state.messages.append({"role": "assistant", "content": full})
        st.rerun()


if __name__ == "__main__":
    import subprocess
    import sys

    import streamlit.runtime as st_runtime

    # `uv run app.py` / `python app.py` are not inside Streamlit — re-invoke properly.
    if not st_runtime.exists():
        sys.exit(
            subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]])
        )
    main()
