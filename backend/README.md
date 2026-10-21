# Industrial Spare Parts Agentic System

A self-hostable, multi-agent AI system that identifies industrial machinery
(JCB backhoes, CAT excavators, etc.) from technician/drone video and
recommends the correct spare part from a vector-searchable catalogue —
with explicit verification checkpoints and human-in-the-loop correction at
every stage where confidence is insufficient to act autonomously.

## Architecture

```
 Video upload (Streamlit UI)
        |
        v
 FastAPI backend  --upload-->  uploads/ (volume)
        |
        v
 LangGraph orchestrator (src/agents/graph.py)
        |
        +--> Agent 1: Perception   (src/agents/perception_agent.py)
        |      OpenCV frame sampling -> Checkpoint 1 (blur/brightness/res)
        |      Vision LLM / mock YOLO-SAM2 detection -> Checkpoint 2 (OCR confidence)
        |
        +--> Agent 2: RAG / Catalogue (src/agents/rag_agent.py)
        |      ChromaDB similarity search -> Checkpoint 3 (similarity threshold)
        |
        +--> Agent 3: Compatibility (src/agents/compatibility_agent.py)
        |      Spec/serial-range validation -> Checkpoint 4 (conflicting parts)
        |
        +--> Agent 4: Clarification / Dialogue (src/agents/clarification_agent.py)
               HITL question generation + resolution
               Final recommendation -> Checkpoint 5 (structured-output retry)
```

Every node is **idempotent**: it skips recomputation when its output is
already valid. This lets the backend simply re-invoke the graph whenever a
human answer lands (`/api/clarify`) — the graph fast-forwards through
already-satisfied stages and resumes exactly where the checkpoint had
paused it. See the design note at the top of `src/agents/graph.py`.

## Verification checkpoints

| # | Checkpoint | Where | Correction |
|---|---|---|---|
| 1 | Frame quality (blur / brightness / resolution) | `src/vision/video_processor.py` | Auto-discard bad frames; if *all* sampled frames fail, pause and ask for a re-upload |
| 2 | OCR / visual entity confidence < 80% | `src/vision/perception.py`, `src/agents/perception_agent.py` | Secondary zoomed rescan of the crop; if still low, flag `uncertain_fields` (non-blocking, surfaced in chat) |
| 3 | RAG cosine similarity < 0.70 | `src/rag/retriever.py`, `src/agents/rag_agent.py` | Pause and ask a clarifying question to refine the query |
| 4 | Conflicting compatible parts (e.g. standard vs heavy-duty) | `src/agents/compatibility_agent.py` | `status = AWAITING_HUMAN_CLARIFICATION`; graph suspends until the user picks |
| 5 | LLM structured JSON output fails Pydantic validation | `src/agents/llm_utils.py` | Automatic retry (max `MAX_STRUCTURED_OUTPUT_RETRIES`, default 3) with the validation error injected into the next prompt |

## Quickstart (Docker)

```bash
cp .env.example .env   # optionally add ANTHROPIC_API_KEY / OPENAI_API_KEY
docker compose up --build
```

- Frontend (Streamlit): http://localhost:8501
- Backend (FastAPI docs): http://localhost:8000/docs
- Chroma server: http://localhost:8010

The backend container automatically ingests `data/spare_parts_catalog.json`
into ChromaDB on startup (idempotent upsert by `part_id`).

Without `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` set, `VISION_PROVIDER=mock`
runs the system fully offline on a deterministic mock vision client — the
whole pipeline, including all 5 checkpoints, still runs end to end.

For real vision inference at zero API cost, set `VISION_PROVIDER=ollama`
and point `OLLAMA_VISION_MODEL` at any vision-capable model pulled into a
local [Ollama](https://ollama.com) install (e.g. `moondream` for speed,
`qwen2.5vl:7b` for stronger OCR on serial tags/brand plates). Nothing
leaves the machine and no account is required.

## Quickstart (local, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
python -m src.rag.ingest_catalog          # embeds & indexes the catalogue locally
uvicorn src.backend.app:app --reload      # terminal 1
streamlit run src/frontend/app.py         # terminal 2
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

- `test_checkpoint1_frame_quality.py`, `test_checkpoint2_ocr_confidence.py`,
  `test_checkpoint4_compatibility.py`, `test_checkpoint5_structured_output.py`
  run fully offline (no network, no external services).
- `test_checkpoint3_rag_similarity.py` and `test_graph_end_to_end.py`
  download the `all-MiniLM-L6-v2` embedding model on first run and use a
  temporary local Chroma directory; they skip gracefully if unavailable.

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /api/upload-video` | Upload a video (`.mp4/.avi/.mov/.mkv`); starts the async pipeline. Pass `existing_job_id` to re-upload for an existing job (Checkpoint 1 correction). |
| `GET /api/agent/stream/{job_id}` | Server-Sent Events stream of live agent state after every node. |
| `POST /api/clarify` | Answer a pending `ClarificationQuestion` (`job_id`, `question_id`, `answer`); resumes the graph. |
| `POST /api/catalog/search` | Direct RAG search over the catalogue, independent of the video pipeline. |
| `GET /api/jobs/{job_id}` | Poll the full current job state. |

## Configuration

See `.env.example` for every tunable, including all five checkpoint
thresholds (`BLUR_VARIANCE_THRESHOLD`, `OCR_CONFIDENCE_THRESHOLD`,
`RAG_SIMILARITY_THRESHOLD`, `MAX_STRUCTURED_OUTPUT_RETRIES`, etc.).

## Known limitations / scope notes

- Job state is in-memory per backend process (`src/backend/job_store.py`);
  scale-out deployments need a shared store (Redis/Postgres) instead.
- The bounding-box detector is a deterministic mock standing in for a real
  YOLO/SAM2 model (see `src/vision/perception.py`); swap in a real
  `VisionClient` implementation to go beyond the demo.
- Compatibility spec-conflict detection covers the distinguishing fields in
  the sample catalogue (`duty`, `rod_diameter_mm`, `bore_diameter_mm`,
  `seal_material`) — extend `_DISTINGUISHING_SPECS` in
  `src/agents/compatibility_agent.py` for a larger real catalogue.
