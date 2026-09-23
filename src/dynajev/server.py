"""HTTP API for a loaded trunk.

One model, one device, one forward at a time. Requests queue behind a bounded
semaphore; when the queue is full the server answers 429 instead of growing
latency without limit. `/api/ready` is the readiness probe (503 until the
weights are on the device); `/api/health` is liveness plus model metadata.

Environment:
    DYNAJEV_MODEL      Hugging Face id (default Qwen/Qwen3.5-2B)
    DYNAJEV_DTYPE      float32 | bfloat16 | float16 (default: bf16 on CPU, fp16 on CUDA)
    DYNAJEV_HOST/PORT  bind address (default 0.0.0.0:43124)
    DYNAJEV_MAX_QUEUE  requests allowed to wait for the model (default 32)
    DYNAJEV_PREFIX_CACHE  prefilled states kept for reuse (default 16)
    DYNAJEV_CORS       comma-separated allowed origins (default *)
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dynajev.compile import DecideIn, QuestionIn
from dynajev.engine import Dynajev
from dynajev.errors import CompileError

MODEL_ID = os.environ.get("DYNAJEV_MODEL", "Qwen/Qwen3.5-2B")
MAX_QUEUE = int(os.environ.get("DYNAJEV_MAX_QUEUE", "32"))
PREFIX_CACHE = int(os.environ.get("DYNAJEV_PREFIX_CACHE", "16"))
VERSION = "0.2.0"

_model_lock = threading.Lock()
_queue = threading.BoundedSemaphore(MAX_QUEUE)
_state: dict[str, Any] = {"dynajev": None, "error": None, "loading": True, "started": time.time(), "served": 0}


def _load() -> None:
    try:
        from dynajev.trunk import Trunk

        trunk = Trunk.load(MODEL_ID)
        trunk.prefix_cache_size = PREFIX_CACHE
        _state["dynajev"] = Dynajev(trunk)
        _state["error"] = None
    except Exception as exc:  # surface the real loader failure to the client
        _state["error"] = str(exc)
    finally:
        _state["loading"] = False


@asynccontextmanager
async def _lifespan(_: FastAPI):
    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(title="Dynajev", version=VERSION, lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("DYNAJEV_CORS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    dynajev: Dynajev | None = _state["dynajev"]
    trunk = getattr(dynajev, "trunk", None)
    return {
        "loaded": dynajev is not None,
        "loading": bool(_state["loading"]),
        "error": _state["error"],
        "model": MODEL_ID,
        "version": VERSION,
        "hidden_size": None if trunk is None else trunk.hidden_size,
        "vocab_size": None if trunk is None else trunk.vocab_size,
        "device": None if trunk is None else trunk.device,
        "dtype": None if trunk is None else str(next(trunk.model.parameters()).dtype).replace("torch.", ""),
        "uptime_s": round(time.time() - _state["started"], 1),
        "served": _state["served"],
        "prefix_cache": None if trunk is None else trunk.prefix_cache_stats(),
    }


@app.get("/api/ready")
def ready() -> dict[str, Any]:
    if _state["dynajev"] is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "loading")
    return {"ready": True, "model": MODEL_ID}


@app.post("/api/decide")
def decide(req: DecideIn) -> dict[str, Any]:
    dynajev: Dynajev | None = _state["dynajev"]
    if dynajev is None:
        detail = _state["error"] or "The model is still loading."
        raise HTTPException(status_code=503, detail=detail)
    if not _queue.acquire(blocking=False):
        raise HTTPException(status_code=429, detail=f"Queue full ({MAX_QUEUE} waiting). Retry shortly.")
    try:
        with _model_lock:
            result = dynajev.decide(req)
        _state["served"] += 1
        return result
    except CompileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _queue.release()


class BatchIn(BaseModel):
    contexts: list[str]
    questions: dict[str, QuestionIn]
    chunk_rows: int = Field(default=32, ge=1, le=256)
    trace: bool = False
    mode: Literal["auto", "dense", "shared"] = "auto"


@app.post("/api/decide_batch")
def decide_batch(req: BatchIn) -> dict[str, Any]:
    dynajev: Dynajev | None = _state["dynajev"]
    if dynajev is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "The model is still loading.")
    if not _queue.acquire(blocking=False):
        raise HTTPException(status_code=429, detail=f"Queue full ({MAX_QUEUE} waiting). Retry shortly.")
    try:
        with _model_lock:
            result = dynajev.decide_batch(
                req.contexts, {k: v.model_dump() for k, v in req.questions.items()}, req.chunk_rows, req.trace, req.mode
            )
        _state["served"] += len(req.contexts)
        return result
    except CompileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _queue.release()


def main() -> None:
    import uvicorn

    host = os.environ.get("DYNAJEV_HOST", "0.0.0.0")
    port = int(os.environ.get("DYNAJEV_PORT", "43124"))
    uvicorn.run("dynajev.server:app", host=host, port=port, reload=False)
