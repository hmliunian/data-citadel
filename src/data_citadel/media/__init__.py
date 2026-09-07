"""Timestamp-preserving MCAP video sampling."""

from .mcap_reader import inspect_mcap
from .sampling import VideoSampler

__all__ = ["VideoSampler", "inspect_mcap"]
