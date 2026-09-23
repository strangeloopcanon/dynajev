"""Model runtimes behind the `Backend` protocol."""

from __future__ import annotations

import os

from dynajev.backends.base import Backend


def load_backend(model_id: str, device: str | None = None, kind: str | None = None) -> Backend:
    kind = (kind or os.environ.get("DYNAJEV_BACKEND", "hf")).lower()
    if kind == "hf":
        from dynajev.backends.hf import HFBackend

        return HFBackend.load(model_id, device)
    raise ValueError(f"Unknown backend {kind!r}. Only 'hf' is implemented; see docs/details.md for what others need.")


__all__ = ["Backend", "load_backend"]
