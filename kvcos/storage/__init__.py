"""ENGRAM Protocol — Storage backends for .eng files."""

from kvcos.storage.backends import StorageBackend
from kvcos.storage.local import LocalStorageBackend

__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
]
