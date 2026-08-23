from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping
import uuid

from aura_orchestration.executor import PickPlaceExecutor
from aura_planning.schemas import dumps_json, loads_json


DEFAULT_TASK_BRIDGE_DIR = Path(
    os.getenv("AURA_VLA_TASK_DIR", "/tmp/aura-vla-control")
).expanduser()


@dataclass(frozen=True)
class TaskBridgePaths:
    directory: Path
    request: Path
    response: Path
    status: Path


def task_bridge_paths(directory: str | Path | None = None) -> TaskBridgePaths:
    bridge_dir = Path(directory or DEFAULT_TASK_BRIDGE_DIR).expanduser().resolve()
    return TaskBridgePaths(
        directory=bridge_dir,
        request=bridge_dir / "request.json",
        response=bridge_dir / "response.json",
        status=bridge_dir / "status.json",
    )


class IsaacTaskBridge:
    """Consumes JSON plans inside Isaac and invokes the injected robot executor."""

    def __init__(
        self,
        execute_pick_place: Callable[[str, str], Any],
        *,
        directory: str | Path | None = None,
        cleanup_after_task: Callable[[], Any] | None = None,
    ) -> None:
        self.paths = task_bridge_paths(directory)
        self._executor = PickPlaceExecutor(execute_pick_place)
        self._cleanup_after_task = cleanup_after_task
        self._task: asyncio.Task[None] | None = None
        self._last_request_id: str | None = None
        self._last_heartbeat = 0.0
        self._owner_token = uuid.uuid4().hex
        self._owner_path = self.paths.directory / "bridge.owner"

    def start(self) -> "IsaacTaskBridge":
        if self._task is not None and not self._task.done():
            return self
        self.paths.directory.mkdir(parents=True, exist_ok=True)
        owner_tmp = self._owner_path.with_suffix(".tmp")
        owner_tmp.write_text(self._owner_token, encoding="utf-8")
        owner_tmp.replace(self._owner_path)
        if self.paths.request.is_file():
            try:
                existing_request = json.loads(
                    self.paths.request.read_text(encoding="utf-8")
                )
                self._last_request_id = str(
                    existing_request.get("request_id") or ""
                ) or None
            except (OSError, json.JSONDecodeError):
                self._last_request_id = None
        self._write_json(
            self.paths.status,
            {
                "ready": True,
                "state": "idle",
                "updated_at_unix": time.time(),
            },
        )
        self._task = asyncio.ensure_future(self._poll_loop())
        print(f"Isaac task bridge started: {self.paths.directory}")
        return self

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._write_json(
            self.paths.status,
            {
                "ready": False,
                "state": "stopped",
                "updated_at_unix": time.time(),
            },
        )

    def process_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("request_id") or "").strip()
        if not request_id:
            raise ValueError("Task request is missing request_id")
        raw_plan = request.get("plan")
        if not isinstance(raw_plan, Mapping):
            raise ValueError("Task request is missing a JSON plan")
        try:
            execution = loads_json(self._executor.execute(dumps_json(raw_plan)))
        finally:
            if self._cleanup_after_task is not None:
                try:
                    self._cleanup_after_task()
                except Exception as exc:
                    print(
                        "Task-end visualization cleanup failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        return {
            "request_id": request_id,
            "success": bool(execution.get("success", False)),
            "execution": execution,
            "completed_at_unix": time.time(),
        }

    async def _poll_loop(self) -> None:
        from omni.kit.app import get_app

        app = get_app()
        while True:
            try:
                try:
                    owner_token = self._owner_path.read_text(encoding="utf-8").strip()
                except OSError:
                    owner_token = ""
                if owner_token != self._owner_token:
                    await app.next_update_async()
                    continue
                now = time.time()
                if now - self._last_heartbeat >= 1.0:
                    self._write_json(
                        self.paths.status,
                        {
                            "ready": True,
                            "state": "idle",
                            "last_request_id": self._last_request_id,
                            "updated_at_unix": now,
                        },
                    )
                    self._last_heartbeat = now
                if self.paths.request.is_file():
                    request = json.loads(
                        self.paths.request.read_text(encoding="utf-8")
                    )
                    request_id = str(request.get("request_id") or "")
                    if request_id and request_id != self._last_request_id:
                        self._last_request_id = request_id
                        self._write_json(
                            self.paths.status,
                            {
                                "ready": True,
                                "state": "executing",
                                "request_id": request_id,
                                "updated_at_unix": time.time(),
                            },
                        )
                        self._last_heartbeat = time.time()
                        print(f"Isaac task received: {request_id}")
                        try:
                            response = self.process_request(request)
                        except Exception as exc:
                            response = {
                                "request_id": request_id,
                                "success": False,
                                "error": str(exc),
                                "exception_type": type(exc).__name__,
                                "completed_at_unix": time.time(),
                            }
                        self._write_json(self.paths.response, response)
                        self._write_json(
                            self.paths.status,
                            {
                                "ready": True,
                                "state": "idle",
                                "last_request_id": request_id,
                                "last_success": bool(response.get("success", False)),
                                "updated_at_unix": time.time(),
                            },
                        )
                        self._last_heartbeat = time.time()
                await app.next_update_async()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._write_json(
                    self.paths.status,
                    {
                        "ready": True,
                        "state": "error",
                        "error": str(exc),
                        "updated_at_unix": time.time(),
                    },
                )
                await app.next_update_async()

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)


class FileTaskClient:
    """Submits a JSON plan to the Isaac process through the task bridge."""

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        timeout_sec: float = 900.0,
        poll_interval_sec: float = 0.1,
    ) -> None:
        self.paths = task_bridge_paths(directory)
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec

    def is_ready(self) -> bool:
        if not self.paths.status.is_file():
            return False
        try:
            status = json.loads(self.paths.status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        updated_at = float(status.get("updated_at_unix") or 0.0)
        return bool(status.get("ready", False)) and time.time() - updated_at <= 5.0

    def execute(self, plan_json: str | Mapping[str, Any]) -> str:
        if not self.is_ready():
            raise RuntimeError(
                f"Isaac task bridge is not ready: {self.paths.status}"
            )
        plan = loads_json(plan_json)
        request_id = uuid.uuid4().hex
        request = {
            "request_id": request_id,
            "submitted_at_unix": time.time(),
            "plan": plan,
        }
        self.paths.directory.mkdir(parents=True, exist_ok=True)
        IsaacTaskBridge._write_json(self.paths.request, request)

        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            if self.paths.response.is_file():
                try:
                    response = json.loads(
                        self.paths.response.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    response = {}
                if response.get("request_id") == request_id:
                    return dumps_json(response)
            time.sleep(self.poll_interval_sec)
        raise TimeoutError(
            f"Isaac did not finish task {request_id} within {self.timeout_sec:.1f}s"
        )


_active_task_bridge: IsaacTaskBridge | None = None


def start_task_bridge(
    execute_pick_place: Callable[[str, str], Any],
    **kwargs: Any,
) -> IsaacTaskBridge:
    global _active_task_bridge
    if _active_task_bridge is not None:
        _active_task_bridge.stop()
    _active_task_bridge = IsaacTaskBridge(execute_pick_place, **kwargs).start()
    return _active_task_bridge


def stop_task_bridge() -> None:
    global _active_task_bridge
    if _active_task_bridge is not None:
        _active_task_bridge.stop()
        _active_task_bridge = None
