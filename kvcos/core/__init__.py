"""ENGRAM Protocol — Core library: types, parsing, compression, serialization, retrieval."""

from kvcos.core.types import (
    ENGRAM_VERSION,
    AttentionType,
    CacheSection,
    CacheSearchResult,
    CacheStats,
    CompressionMethod,
    EngramMetadata,
    ModelCacheSpec,
    StateExtractionMode,
)
from kvcos.core.manifold_index import IndexEntry, ManifoldIndex
from kvcos.core.retriever import EGRRetriever, RetrievalResponse, RetrievalResult
from kvcos.core.state_extractor import ExtractionResult, MARStateExtractor, SVDProjection

__all__ = [
    # Types
    "ENGRAM_VERSION",
    "AttentionType",
    "CacheSection",
    "CacheSearchResult",
    "CacheStats",
    "CompressionMethod",
    "EngramMetadata",
    "ModelCacheSpec",
    "StateExtractionMode",
    # Manifold index
    "IndexEntry",
    "ManifoldIndex",
    # Retriever
    "EGRRetriever",
    "RetrievalResponse",
    "RetrievalResult",
    # State extraction (MAR)
    "ExtractionResult",
    "MARStateExtractor",
    "SVDProjection",
]
