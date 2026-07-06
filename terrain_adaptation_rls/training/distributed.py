"""Small helpers for distributed training environment state."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


@dataclass(frozen=True)
class DistributedContext:
    """Distributed rank information from torchrun-compatible environment vars."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DistributedContext":
        env = os.environ if env is None else env
        return cls(
            rank=int(env.get("RANK", "0")),
            local_rank=int(env.get("LOCAL_RANK", "0")),
            world_size=int(env.get("WORLD_SIZE", "1")),
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0
