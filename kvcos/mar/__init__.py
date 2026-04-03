"""ENGRAM Protocol — MAR (Manifold Attention Retrieval)

Backward compatibility re-exports. All classes have moved to kvcos.core.
Import from kvcos.core directly for new code.
"""

from kvcos.core.manifold_index import IndexEntry, ManifoldIndex
from kvcos.core.retriever import EGRRetriever, RetrievalResponse, RetrievalResult
from kvcos.core.state_extractor import ExtractionResult, MARStateExtractor, SVDProjection

__all__ = [
    "IndexEntry",
    "ManifoldIndex",
    "EGRRetriever",
    "RetrievalResponse",
    "RetrievalResult",
    "ExtractionResult",
    "MARStateExtractor",
    "SVDProjection",
]
