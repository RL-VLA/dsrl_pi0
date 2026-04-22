"""Demo: batched scoring with overlapping physics sim.

Assumes a Robometer eval server is running at http://localhost:8000:

    uv run python robometer/evals/eval_server.py server_url=0.0.0.0 server_port=8000

Run:

    python examples/batch_inference.py
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from robometer_client import RobometerClient, VideoSample

SERVER_URL = "http://localhost:8000"


def fake_physics_sim(batch_size: int = 4, T: int = 24, hw: int = 224) -> List[VideoSample]:
    """Stand-in for your real sim: returns a batch of random trajectories.

    In the real thing, `frames` would come from your renderer.
    """
    rng = np.random.default_rng()
    samples = []
    for i in range(batch_size):
        frames = rng.integers(0, 255, size=(T, hw, hw, 3), dtype=np.uint8)
        samples.append(
            VideoSample(
                frames=frames,
                task=f"pick up the red block — rollout {i}",
                id=f"traj_{i}",
            )
        )
    # Pretend the sim actually took some wall-clock time:
    time.sleep(0.5)
    return samples


def main() -> None:
    with RobometerClient(SERVER_URL, max_concurrent_requests=4) as client:
        print("health:", client.health())

        t0 = time.perf_counter()

        # Round 1: collect trajs, submit for scoring (doesn't block)
        trajs_a = fake_physics_sim(batch_size=4)
        fut_a = client.submit(trajs_a)

        # Round 2: the server is scoring round 1 on a background thread while
        # this sim runs on the main thread. Wall-clock is max(HTTP, sim), not sum.
        trajs_b = fake_physics_sim(batch_size=4)
        fut_b = client.submit(trajs_b)

        # Collect
        results_a = fut_a.result()
        results_b = fut_b.result()

        elapsed = time.perf_counter() - t0

        for label, results in (("A", results_a), ("B", results_b)):
            print(f"\nBatch {label}:")
            for r in results:
                succ_last = float(r.success[-1]) if r.success is not None else None
                print(
                    f"  {r.id}: progress[-1]={float(r.progress[-1]):.3f} "
                    f"T={len(r.progress)} success[-1]={succ_last}"
                )

            # Best-of-N by final progress:
            best = max(results, key=lambda x: float(x.progress[-1]))
            print(f"  best of N: {best.id} ({float(best.progress[-1]):.3f})")

        print(f"\nTotal wall-clock: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
