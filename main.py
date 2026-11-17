from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Body, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.session import TrainingSession, training_session
from backend.config import ENVIRONMENT_VERSION, MODEL_VERSION, REWARD_VERSION


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
Seed = Annotated[int, Field(ge=0, le=2_147_483_647)]
EvaluationSeed = Annotated[int, Field(ge=0, le=4_294_967_295)]


class AdvanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: int = Field(default=1, ge=1, le=2048)


class TrainRequest(AdvanceRequest):
    steps: int = Field(default=256, ge=1, le=2048)


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paused: bool | None = None
    intro: bool | None = None
    mode: Literal["train", "eval"] | None = None
    level: int | None = Field(default=None, ge=1, le=6)
    autoCurriculum: bool | None = None
    fullSurvival: bool | None = None
    seed: Seed | None = None
    newEpisode: bool | None = None


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episodes: int = Field(default=8, ge=1, le=32)
    seeds: list[EvaluationSeed] | None = Field(default=None, min_length=1, max_length=32)

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, seeds: list[int] | None) -> list[int] | None:
        if seeds is not None and len(set(seeds)) != len(seeds):
            raise ValueError("Evaluation seeds must be distinct.")
        return seeds


def create_app(session: TrainingSession | None = None) -> FastAPI:
    service = session if session is not None else training_session
    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        try:
            yield
        finally:
            service.close()

    application = FastAPI(title="AI Snake Training Room", version="3.1.0", lifespan=lifespan)
    application.state.session = service
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.middleware("http")
    async def prevent_stale_client_state(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def invoke(method, *args, **kwargs) -> dict[str, object]:
        try:
            return method(*args, **kwargs)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/", include_in_schema=False)
    def read_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @application.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "modelVersion": MODEL_VERSION,
                "environmentVersion": ENVIRONMENT_VERSION, "simulationAuthority": "python",
                "rewardVersion": REWARD_VERSION, "contactSchema": 1}

    @application.get("/api/state")
    def get_state() -> dict[str, object]:
        return service.get_state()

    @application.post("/api/tick")
    def tick(options: AdvanceRequest | None = Body(default=None)) -> dict[str, object]:
        return invoke(service.tick, steps=options.steps if options else 1)

    @application.post("/api/control")
    def control(options: ControlRequest) -> dict[str, object]:
        return invoke(service.control, options.model_dump(exclude_none=True))

    @application.post("/api/train")
    def train(options: TrainRequest | None = Body(default=None)) -> dict[str, object]:
        return invoke(service.train_steps, steps=options.steps if options else 256)

    @application.post("/api/evaluate")
    def evaluate(options: EvaluationRequest | None = Body(default=None)) -> dict[str, object]:
        settings = options or EvaluationRequest()
        return invoke(service.evaluate, episodes=settings.episodes, seeds=settings.seeds)

    @application.post("/api/reset")
    def reset_training() -> dict[str, object]:
        return invoke(service.reset_training)

    @application.post("/api/save-weights")
    def save_weights() -> dict[str, object]:
        return invoke(service.save_manual_checkpoint)

    @application.post("/api/load-weights")
    def load_weights() -> dict[str, object]:
        return invoke(service.load_saved_or_best_checkpoint)

    @application.post("/api/load-ready-weights")
    def load_ready_weights() -> dict[str, object]:
        return invoke(service.load_ready_weights)

    @application.post("/api/check-best")
    def check_best() -> dict[str, object]:
        return invoke(service.check_best_policy)

    return application


app = create_app()
