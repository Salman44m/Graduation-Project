# PromptEvo (Bayyen) AI Red-Teaming Framework

## Project Overview

**PromptEvo** is an advanced, automated, graph-based AI red-teaming and jailbreaking framework designed to evaluate and enhance the security of Large Language Models (LLMs). Built on top of LangGraph, it features an Enterprise Command Center (Streamlit Dashboard) and a robust REST API layer (FastAPI) for CI/CD integration. 

PromptEvo orchestrates complex interactions between attacker models and target models, leveraging a "war room" style architecture to systematically uncover vulnerabilities and generate defensive patches in real time.

## System Requirements

- **Python**: `>= 3.11`
- **Package Manager**: `pip` or `uv`
- **Dependencies**: See `requirements.txt` or `pyproject.toml` (includes `langgraph`, `streamlit`, `fastapi`, `anthropic`, `langchain-google-genai`, `openai`, `faiss-cpu`, etc.).

### API Keys
You will need relevant API keys depending on the models configured for the attacker, target, and judge. Define these in your `.env` file:
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY` (for Gemini quotas)
- `OPENAI_API_KEY`
- `GROQ_API_KEY` (if using Groq endpoints)

*(Refer to `.env.example` in the root directory for a full list of supported environment variables.)*

## Running the System

Ensure you have installed the required dependencies from `requirements.txt` or using the `pyproject.toml`:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### 1. Streamlit War Room Dashboard
To launch the cinematic, real-time command center interface:

```bash
streamlit run dashboard.py
```
This interface provides a live view of the audit sessions, node-by-node execution updates, and human-in-the-loop (HITL) capabilities.

### 2. Enterprise REST API
To launch the FastAPI layer for programmatic access and CI/CD security gate integration:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
The API provides endpoints such as `POST /api/v1/audit` to start an audit and `GET /api/v1/audit/{session_id}/stream` for server-sent events.

## Project Structure

A brief overview of the main directories in the framework:

- `agents/`: Contains the core agents responsible for the red-teaming lifecycle (e.g., Attacker, Target, Judge).
- `core/`: Houses the foundational LangGraph orchestrator, state definitions, and system constants.
- `intelligence/`: Modules governing the adaptive learning and strategic planning of the red-teaming engine.
- `memory/`: Implements thread-safe session memory, checkpoints, and vector databases (e.g., FAISS) for long-term intelligence gathering.
- `schemas/`: Defines the strict Pydantic data models used for internal state management and API validation.
- `evaluators/`: Assessment modules used to calculate Risk & Harm Scores (RAHS), cooperation metrics, and success probabilities.
- `remediation/`: Contains logic for dynamically generating and applying defensive patches to mitigate identified vulnerabilities.
- `hitl/`: Human-In-The-Loop integration components allowing manual overrides and approvals during an audit.
- `infra/`: Shared infrastructure code handling database connections, observability/logging, and security middleware.
