"""Training utilities for single-process and DDP runs."""

from .distributed import DistributedContext

__all__ = ["DistributedContext"]
