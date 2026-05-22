"""Standalone FastAPI service that exposes the SPLADE sparse encoder over HTTP.

Deploy this on the GPU machine.  Client machines set ``SPLADE_API_URL`` to
point at this server and :class:`~src.clustering.splade_encoder.SPLADEEncoder`
will call it instead of loading the model locally.

Running
-------
::

    # development
    uvicorn splade_api:app --reload --port 8002

    # production (single worker — model holds GPU; recycle to bound RSS)
    gunicorn -w 1 -k uvicorn.workers.UvicornWorker \\
        --max-requests 500 --max-requests-jitter 50 \\
        -b 0.0.0.0:8002 "splade_api:app" \\
        --graceful-timeout 30 --timeout 120

Environment variables
---------------------
Same model / cache settings as the main app — ``SPLADEConfigurations`` in
``src/constants.py`` controls which model variant is loaded.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict

import scipy.sparse
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.clustering.splade_encoder import SPLADEEncoder
from src.constants import SPLADEConfigurations
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    enc = SPLADEEncoder()
    if enc.is_available:
        logger.info(f"[SPLADE API] Model ready — device={enc.device}, model={enc._model_name}")
    else:
        logger.warning("[SPLADE API] SPLADE model unavailable — /api/splade/encode/ will return 503")
    yield
    SPLADEEncoder.release()
    logger.info("[SPLADE API] Model released on shutdown")


app = FastAPI(
    title="SPLADE Encoding API",
    description="GPU-backed SPLADE sparse encoding service for the issue-grouping pipeline.",
    version="1.0.0",
    docs_url="/api",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)


class SpladeEncodeRequest(BaseModel):
    texts: list[str] = Field(description="List of text strings to encode with SPLADE")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/api")


@app.get("/api/health/", status_code=200)
async def health() -> Dict:
    """Liveness / readiness check."""
    enc = SPLADEEncoder()
    return {
        "status": "ok" if enc.is_available else "degraded",
        "model": enc._model_name,
        "device": enc.device,
    }


@app.post("/api/splade/encode/", status_code=200)
async def splade_encode(request: SpladeEncodeRequest) -> Dict:
    if not request.texts:
        return ORJSONResponse(status_code=400, content={"status": 400, "error": "texts list cannot be empty"})

    enc = SPLADEEncoder()
    if not enc.is_available:
        return ORJSONResponse(status_code=503, content={"status": 503, "error": "SPLADE model is not available"})

    loop = asyncio.get_event_loop()
    chunk_size = max(1, SPLADEConfigurations.max_inference_chunk)
    texts = request.texts

    if len(texts) <= chunk_size:
        vecs = await loop.run_in_executor(None, enc.encode, texts)
    else:
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
        parts = []
        for chunk in chunks:
            part = await loop.run_in_executor(None, enc.encode, chunk)
            if part is None:
                return ORJSONResponse(status_code=500, content={"status": 500, "error": "Encoding failed"})
            parts.append(part)
        vecs = scipy.sparse.vstack(parts, format="csr")

    if vecs is None:
        return ORJSONResponse(status_code=500, content={"status": 500, "error": "Encoding failed"})

    return {
        "status": 200,
        "model": enc._model_name,
        "vocab_size": int(vecs.shape[1]),
        "count": int(vecs.shape[0]),
        "shape": [int(vecs.shape[0]), int(vecs.shape[1])],
        "indptr": vecs.indptr.tolist(),
        "indices": vecs.indices.tolist(),
        "data": vecs.data.tolist(),
    }
