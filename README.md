# Streamlit UI for local Ollama

This is a Streamlit UI for local Ollama. It allows you to control the Ollama service, list available models, and chat with the models.

## Linux

Install uv

```bash
curl -fsSL https://astral.sh/uv/install.sh | sh
```

Clone the repository

```bash
git clone https://github.com/ALeliuhin/ollama-streamlit-app.git
cd ollama-streamlit-app
```

Install dependencies

```bash
uv sync
```

Running

```bash
uv run -m src.app
```

Add alias to `.bashrc`

```bash
printf "\nalias ollama-ui='cd <path_to_ollama-streamlit-app> && uv run -m src.app'\n" >> ~/.bashrc
source ~/.bashrc
```

## Windows

Install uv (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone the repository

```powershell
git clone https://github.com/ALeliuhin/ollama-streamlit-app.git
cd ollama-streamlit-app
```

Install dependencies

```powershell
uv sync
```

Running

```powershell
uv run -m src.app
```
