# Agentic ACE — Spare Parts Identification

An agentic pipeline that identifies industrial spare parts from camera footage. A simulated Crazyflie drone in Webots spots an Epson SCARA T6 arm on a sorting line, crops it, and sends the image to a multi-agent backend that matches it against a spare-parts catalogue.

**Demo video:** [`webots_sim/demo/agentic_ace_webots_final_demo.mp4`](webots_sim/demo/agentic_ace_webots_final_demo.mp4)

## Layout

| Folder | What it is |
|---|---|
| `webots_sim/` | Webots world (`worlds/industrial_example.wbt`): an Epson SCARA T6 sorts apples and oranges while a Crazyflie runs YOLO-World to detect the arm. `controllers/crazyflie/spare_parts_agent_bridge.py` posts crops to the backend. |
| `backend/` | FastAPI + Streamlit agentic backend (perception → visual retrieval → RAG over ChromaDB → verification). Exposes `POST /api/upload-image`. See `backend/README.md`. |

## Running

1. **Backend**
   ```bash
   cd backend
   python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ollama pull qwen2.5vl:3b
   OLLAMA_VISION_MODEL=qwen2.5vl:3b uvicorn src.backend.app:app --port 8000
   ```
2. **Simulation**: open `webots_sim/worlds/industrial_example.wbt` in Webots (R2023b+).
   - `CF_AUTOPILOT=1` flies a scripted route.
   - `CF_RECORD_DIR=<dir>` and `CF_RECORD_SECONDS=<n>` record `scene.mp4` and `hud.mp4` against the sim clock.
   - YOLO / CLIP weights are not committed. Ultralytics downloads `yolov8s-worldv2.pt` on first run, and CLIP downloads `ViT-B-32.pt`.

Large raw datasets (`backend/data/mvid_raw`), model weights and runtime uploads are left out of the repo.
