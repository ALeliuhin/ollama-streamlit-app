# Streamlit UI for local Ollama

This is a Streamlit UI for local Ollama. It allows you to control the Ollama service, list available models, and chat with the models.

# Installation

Install uv
```bash
curl -fsSL https://astral.sh/uv/install.sh | sh
```

Clone the repository
```bash
git clone https://github.com/ALeliuhin/ollama-streamlit-app.git
```
Install dependencies
```bash
uv sync
```

# Running

```bash
uv run -m src.app
```