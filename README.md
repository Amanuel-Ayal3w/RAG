# RAG Fullstack (Next.js + FastAPI + OpenAI + pgvector)

This workspace is split into:

- `frontend/`: Next.js modern chat UI
- `backend/`: FastAPI RAG API with OpenAI and PostgreSQL `pgvector`

## 1) Prepare local PostgreSQL + pgvector

Create a local database/user that matches `backend/.env` defaults:

```bash
sudo -u postgres psql -c "CREATE USER rag WITH PASSWORD 'rag';"
sudo -u postgres psql -c "CREATE DATABASE ragdb OWNER rag;"
sudo -u postgres psql -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

If `CREATE EXTENSION vector` fails, install pgvector in your local PostgreSQL first, then rerun the command.

## 2) Run backend (FastAPI)

```bash
cd backend
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Set `OPENAI_API_KEY` in `backend/.env`.

## 3) Run frontend (Next.js)

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

## API endpoints

- `POST /api/v1/ingest/documents`
	- Form-data key: `files` (multiple allowed)
	- Supported: `.txt`, `.md`, `.pdf`, `.docx`

- `POST /api/v1/chat`
	- Body:
		```json
		{
			"conversation_id": null,
			"message": "What does our policy say about refunds?"
		}
		```

## Conversation memory

The backend keeps a **sliding window** of recent conversation turns (`MEMORY_WINDOW`, default `8`) so each chat request includes recent context plus retrieved vector knowledge.
