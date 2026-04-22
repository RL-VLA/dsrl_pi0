"""Multipart payload builder + response parser for /evaluate_batch_npy.

Reproduces the wire contract defined by robometer/evals/eval_server.py and
robometer/evals/eval_utils.py. Kept as a thin internal module so the client
stays dependency-free of the robometer package itself.
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from robometer_client.types import ScoreResult, VideoSample

_NPY_MEDIA_TYPE = "application/octet-stream"

# httpx files value = (filename, content, content_type)
HttpxFile = Tuple[str, bytes, str]
# httpx files param = list of (field_name, file_tuple)
HttpxFilesList = List[Tuple[str, HttpxFile]]


def _normalize_frames(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim != 4:
        raise ValueError(f"frames must be 4D (T,H,W,C) or (T,C,H,W); got shape {arr.shape}")
    # (T, C, H, W) -> (T, H, W, C) when C is 1 or 3 in axis 1 and not in axis -1
    if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _sample_id(sample: VideoSample, index: int) -> str:
    return sample.id if sample.id is not None else str(index)


def build_multipart(
    samples: Sequence[VideoSample],
    *,
    use_frame_steps: bool,
) -> Tuple[HttpxFilesList, Dict[str, str]]:
    """Build the multipart (files, data) payload for POST /evaluate_batch_npy.

    Each sample becomes one JSON form field (`sample_{i}`) referencing its
    frames by file key (`sample_{i}_trajectory_frames`), with the frames
    themselves uploaded as .npy blobs.
    """
    if not samples:
        raise ValueError("samples must be non-empty")

    files: HttpxFilesList = []
    data: Dict[str, str] = {"use_frame_steps": "true" if use_frame_steps else "false"}

    for i, sample in enumerate(samples):
        frames = _normalize_frames(sample.frames)
        T = int(frames.shape[0])
        file_key = f"sample_{i}_trajectory_frames"

        buf = io.BytesIO()
        np.save(buf, frames)
        files.append((file_key, (f"{file_key}.npy", buf.getvalue(), _NPY_MEDIA_TYPE)))

        sample_json = {
            "sample_type": "progress",
            "trajectory": {
                "frames": {"__numpy_file__": file_key},
                "frames_shape": list(frames.shape),
                "task": sample.task,
                "id": _sample_id(sample, i),
                "metadata": {"subsequence_length": T},
                "video_embeddings": None,
            },
        }
        data[f"sample_{i}"] = json.dumps(sample_json)

    return files, data


def parse_response(
    resp_json: Dict[str, Any],
    samples: Sequence[VideoSample],
) -> List[ScoreResult]:
    """Parse /evaluate_batch_npy response into positionally-aligned ScoreResults."""
    outputs_progress = resp_json.get("outputs_progress") or {}
    progress_list = outputs_progress.get("progress_pred") or []

    # outputs_success may be top-level OR nested under outputs_progress depending
    # on server code path; tolerate both.
    outputs_success = resp_json.get("outputs_success") or outputs_progress.get("outputs_success") or {}
    success_list = outputs_success.get("success_probs") or []

    if len(progress_list) != len(samples):
        raise RuntimeError(
            f"Server returned {len(progress_list)} progress arrays for "
            f"{len(samples)} input samples — positional alignment broken."
        )

    results: List[ScoreResult] = []
    for i, sample in enumerate(samples):
        progress = np.asarray(progress_list[i], dtype=np.float32)

        succ: Optional[np.ndarray]
        if i < len(success_list) and success_list[i]:
            succ = np.asarray(success_list[i], dtype=np.float32)
        else:
            succ = None

        results.append(
            ScoreResult(
                id=_sample_id(sample, i),
                progress=progress,
                success=succ,
            )
        )
    return results
