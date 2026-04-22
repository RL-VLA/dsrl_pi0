from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VideoSample:
    """One video + task to be scored.

    frames: 4D numpy array. Accepted layouts: (T, H, W, C) or (T, C, H, W).
        Any numeric dtype is clamped to [0, 255] and cast to uint8 before upload.
    task: natural-language task instruction, e.g. "Pick up the red block".
    id: optional stable identifier. Echoed back in the matching ScoreResult.
        If None, the result is keyed by its position in the batch ("0", "1", ...).
    """

    frames: np.ndarray
    task: str
    id: Optional[str] = None


@dataclass
class ScoreResult:
    """Per-video score returned by the server.

    id: identifier of the input VideoSample (or its positional index if id was None).
    progress: (T,) float32 array of per-frame progress in [0, 1].
    success: (T,) float32 array of per-frame success probabilities,
        or None if the loaded model has no success head.
    """

    id: str
    progress: np.ndarray
    success: Optional[np.ndarray]
