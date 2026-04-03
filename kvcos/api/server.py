"""
ENGRAM Protocol — ENGRAM Server


FastAPI application factory with lifespan management.
Initializes storage, index, extractor, and retriever on startup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from kvcos.api import routes
from kvcos.core.config import get_config
from kvcos.core.serializer import EngramSerializer
from kvcos.core.types import ENGRAM_VERSION, StateExtractionMode
from kvcos.core.manifold_index import ManifoldIndex
from kvcos.core.retriever import EGRRetriever
from kvcos.core.state_extractor import MARStateExtractor
from kvcos.storage.local import LocalStorageBackend

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ENGRAM components on startup, clean up on shutdown."""
    config = get_config()

    # Initialize storage backend
    storage = LocalStorageBackend(data_dir=config.data_dir)

    # Initialize EGR manifold index
    index_path = config.index_dir / "egr.faiss"
    index = ManifoldIndex(dim=config.state_vec_dim, index_path=index_path)

    # Initialize state extractor
    extractor = MARStateExtractor(
        mode=StateExtractionMode.SVD_PROJECT,
        rank=config.state_vec_dim,
    )

    # Initialize retriever
    serializer = EngramSerializer()
    retriever = EGRRetriever(
        extractor=extractor,
        index=index,
        storage=storage,
        serializer=serializer,
    )

    # Wire into route handlers
    routes._storage = storage
    routes._index = index
    routes._retriever = retriever

    logger.info("ENGRAM v%s started", ENGRAM_VERSION)
    logger.info("  Storage:  %s (%d entries)", config.data_dir, storage.stats()["total_entries"])
    logger.info("  Index:    %s (%d vectors, dim=%d)", config.index_dir, index.n_entries, config.state_vec_dim)
    logger.info("  Backend:  %s", config.backend.value)

    yield

    # Shutdown: persist index
    try:
        index.save(index_path)
        logger.info("Index saved to %s", index_path)
    except Exception as e:
        logger.warning("Failed to save index: %s", e)

    # Clear route references
    routes._storage = None
    routes._index = None
    routes._retriever = None

    logger.info("ENGRAM shutdown complete")


def create_app() -> FastAPI:
    """Create the ENGRAM FastAPI application."""
    app = FastAPI(
        title="ENGRAM Protocol API",
        description="ENGRAM Protocol: Cognitive state, persisted.",
        version=ENGRAM_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(routes.router)
    return app


def main() -> None:
    """Entry point for `engram-server` console script."""
    import uvicorn

    config = get_config()
    application = create_app()
    uvicorn.run(
        application,
        host=config.host,
        port=config.port,
        log_level="info",
    )


def _get_app() -> FastAPI:
    """Lazy app factory for `uvicorn kvcos.api.server:app`.

    Defers create_app() until the attribute is actually accessed,
    avoiding side effects on module import.
    """
    return create_app()


def __getattr__(name: str) -> FastAPI:
    if name == "app":
        return _get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
