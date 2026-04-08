"""
FraudDetectionEnv — HTTP client for the fraud detection environment.

Works both as an async client and via a synchronous context-manager wrapper:

    # Async
    env = FraudDetectionEnv("http://localhost:8000")
    obs = await env.reset("easy")
    result = await env.step(action)

    # Sync
    with FraudDetectionEnv("http://localhost:8000").sync() as env:
        obs = env.reset("easy")
        result = env.step(action)
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

import requests

from models import FraudAction, FraudObservation, FraudState


# ── Shared result type ─────────────────────────────────────────────────────────

@dataclass
class StepResult:
    observation: FraudObservation
    reward: float
    done: bool


# ══════════════════════════════════════════════════════════════════════════════
#  Async client (primary)
# ══════════════════════════════════════════════════════════════════════════════

class FraudDetectionEnv:
    """
    Async HTTP client for the fraud detection environment.

    Parameters
    ----------
    base_url : str
        Root URL of the running server (e.g. "http://localhost:8000").
    timeout  : int
        Request timeout in seconds.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ── Payload / parse helpers ────────────────────────────────────────────

    def _step_payload(self, action: FraudAction) -> dict:
        return action.model_dump()

    def _parse_observation(self, payload: dict) -> FraudObservation:
        return FraudObservation(**payload)

    def _parse_result(self, payload: dict) -> StepResult:
        obs = self._parse_observation(payload)
        return StepResult(
            observation=obs,
            reward=payload.get("reward") or 0.0,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> FraudState:
        return FraudState(**payload)

    # ── Async API ──────────────────────────────────────────────────────────

    async def reset(self, task_level: str = "easy") -> FraudObservation:
        """Start a fresh episode. Returns the first FraudObservation."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._session.post(
                f"{self.base_url}/reset",
                json={"task_level": task_level},
                timeout=self.timeout,
            ),
        )
        response.raise_for_status()
        return self._parse_observation(response.json())

    async def step(self, action: FraudAction) -> StepResult:
        """Submit one action. Returns StepResult(observation, reward, done)."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._session.post(
                f"{self.base_url}/step",
                json=self._step_payload(action),
                timeout=self.timeout,
            ),
        )
        response.raise_for_status()
        return self._parse_result(response.json())

    async def get_state(self) -> FraudState:
        """Fetch current episode metadata."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._session.get(
                f"{self.base_url}/state",
                timeout=self.timeout,
            ),
        )
        response.raise_for_status()
        return self._parse_state(response.json())

    async def health(self) -> dict:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._session.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            ),
        )
        response.raise_for_status()
        return response.json()

    # ── Sync wrapper factory ───────────────────────────────────────────────

    def sync(self) -> "SyncFraudDetectionEnv":
        """Return a synchronous wrapper around this client."""
        return SyncFraudDetectionEnv(self)

    def close(self) -> None:
        self._session.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Synchronous wrapper
# ══════════════════════════════════════════════════════════════════════════════

class SyncFraudDetectionEnv:
    """
    Synchronous context-manager wrapper around FraudDetectionEnv.

    Usage
    -----
    with FraudDetectionEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset("easy")
        while not obs.done:
            action = ...
            result = env.step(action)
            obs = result.observation
    """

    def __init__(self, async_env: FraudDetectionEnv) -> None:
        self._env = async_env

    def __enter__(self) -> "SyncFraudDetectionEnv":
        return self

    def __exit__(self, *args: object) -> None:
        self._env.close()

    def _run(self, coro):  # type: ignore[no-untyped-def]
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def reset(self, task_level: str = "easy") -> FraudObservation:
        return self._run(self._env.reset(task_level))

    def step(self, action: FraudAction) -> StepResult:
        return self._run(self._env.step(action))

    def get_state(self) -> FraudState:
        return self._run(self._env.get_state())

    def health(self) -> dict:
        return self._run(self._env.health())
