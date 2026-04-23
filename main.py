from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.session import training_session


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Snake DQN API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def get_state() -> dict[str, object]:
    return training_session.get_state()


@app.post("/api/tick")
def tick() -> dict[str, object]:
    return training_session.tick()


@app.post("/api/reset")
def reset_training() -> dict[str, object]:
    return training_session.reset_training()


@app.post("/api/save-weights")
def save_weights() -> dict[str, object]:
    return training_session.save_manual_checkpoint()


@app.post("/api/load-weights")
def load_weights() -> dict[str, object]:
    return training_session.load_saved_or_best_checkpoint()
