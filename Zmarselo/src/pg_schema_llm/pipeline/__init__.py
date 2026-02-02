"""
Pipeline orchestration package for schema discovery.

This package groups the individual pipeline stages used for
ground-truth extraction, profiling, schema inference, and evaluation.
Pipeline stages are intentionally not imported at package level to
avoid loading heavy or optional dependencies by default.

Each stage should be imported explicitly from its corresponding
module to maintain modularity and reduce startup overhead.
"""


__all__ = []
# Empty __all__ to avoid importing anything by default