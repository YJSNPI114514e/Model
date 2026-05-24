"""GRIM: Geometric RKHS Integrative Model."""

from grim.config import GRIMConfig

__all__ = ["GRIMConfig", "GRIM"]


def __getattr__(name: str):
    if name == "GRIM":
        from grim.model import GRIM

        return GRIM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
