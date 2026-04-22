"""RobometerClient — batch video scoring with a sync API and Future-based overlap.

Typical usage (sync sim interleaved with inference)::

    from robometer_client import RobometerClient, VideoSample

    with RobometerClient("http://localhost:8000") as client:
        trajs_a = run_physics_sim()          # your existing blocking sim
        fut_a = client.submit(trajs_a)        # returns immediately

        trajs_b = run_physics_sim()           # runs while fut_a is in flight
        fut_b = client.submit(trajs_b)

        results_a = fut_a.result()            # list[ScoreResult]
        results_b = fut_b.result()

        # best-of-N by final progress value:
        best = max(results_a, key=lambda r: float(r.progress[-1]))
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import httpx

from robometer_client._payload import build_multipart, parse_response
from robometer_client.types import ScoreResult, VideoSample

_RETRYABLE_STATUS = {502, 503, 504}


class RobometerClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 300.0,
        max_concurrent_requests: int = 8,
        max_retries: int = 0,
    ) -> None:
        """
        base_url: e.g. "http://localhost:8000" — the Robometer eval server URL.
        timeout: per-request HTTP timeout in seconds.
        max_concurrent_requests: upper bound on in-flight submit() batches.
            Also sizes the HTTP connection pool.
        max_retries: retries for network errors / 5xx responses. 0 = fail fast.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.max_retries = max_retries

        limits = httpx.Limits(
            max_connections=max_concurrent_requests,
            max_keepalive_connections=max_concurrent_requests,
        )
        self._http = httpx.Client(timeout=timeout, limits=limits)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_requests,
            thread_name_prefix="robometer-client",
        )
        self._closed = False

    def __enter__(self) -> "RobometerClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        self._http.close()

    # ---- health / info ----

    def health(self) -> Dict[str, Any]:
        r = self._get("/health")
        return r.json()

    def model_info(self) -> Dict[str, Any]:
        r = self._get("/model_info")
        return r.json()

    def gpu_status(self) -> Dict[str, Any]:
        r = self._get("/gpu_status")
        return r.json()

    # ---- scoring ----

    def score(
        self,
        samples: Sequence[VideoSample],
        *,
        use_frame_steps: bool = False,
    ) -> List[ScoreResult]:
        """Score a batch of videos synchronously. Blocks until the server responds.

        Results are positionally aligned with `samples`.
        """
        if not samples:
            return []
        files, data = build_multipart(samples, use_frame_steps=use_frame_steps)
        resp = self._post("/evaluate_batch_npy", files=files, data=data)
        return parse_response(resp.json(), samples)

    def submit(
        self,
        samples: Sequence[VideoSample],
        *,
        use_frame_steps: bool = False,
    ) -> Future:
        """Submit a batch and get a Future back.

        The HTTP request runs on an internal worker thread so the caller can
        keep running sim work on the main thread. Call `.result()` on the
        Future when the scores are needed.
        """
        if self._closed:
            raise RuntimeError("Client is closed")
        # Materialize samples eagerly so callers can mutate the source list.
        frozen = list(samples)
        return self._executor.submit(
            self.score, frozen, use_frame_steps=use_frame_steps
        )

    # ---- internals ----

    def _get(self, path: str) -> httpx.Response:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.get(url)
                if r.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    last_exc = _http_error(r)
                    continue
                _raise_for_status(r)
                return r
            except httpx.TransportError as e:
                last_exc = e
                if attempt >= self.max_retries:
                    raise
        assert last_exc is not None
        raise last_exc

    def _post(self, path: str, *, files, data) -> httpx.Response:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.post(url, files=files, data=data)
                if r.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    last_exc = _http_error(r)
                    continue
                _raise_for_status(r)
                return r
            except httpx.TransportError as e:
                last_exc = e
                if attempt >= self.max_retries:
                    raise
        assert last_exc is not None
        raise last_exc


def _raise_for_status(r: httpx.Response) -> None:
    if r.is_success:
        return
    raise _http_error(r)


def _http_error(r: httpx.Response) -> httpx.HTTPStatusError:
    body = r.text[:1000] if r.text else ""
    msg = (
        f"Robometer server returned {r.status_code} {r.reason_phrase} "
        f"for {r.request.method} {r.request.url}\n"
        f"Body (truncated): {body}"
    )
    return httpx.HTTPStatusError(msg, request=r.request, response=r)
