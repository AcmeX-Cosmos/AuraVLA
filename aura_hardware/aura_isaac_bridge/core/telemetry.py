"""File-backed telemetry for the Isaac Sim to ROS 2 bridge.

The task executor runs inside Isaac Sim's Python process and must not import
``rclpy``.  It therefore publishes the latest structured event through an
atomic JSON file.  The ROS bridge consumes that file and republishes it on a
topic.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from threading import Lock


DEFAULT_TRANSPORT_TRACKING_FILE = "/tmp/aura-vla-control/transport_tracking.json"


def _json_default(value):
    """Convert common runtime values without importing NumPy in the bridge."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return str(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


class TransportTelemetryWriter:
    """Atomically publish the latest transport tracking event."""

    def __init__(self, path: str | os.PathLike | None = None):
        configured = path or os.environ.get(
            "AURA_TRANSPORT_TRACKING_FILE",
            DEFAULT_TRANSPORT_TRACKING_FILE,
        )
        self.path = Path(configured).expanduser()
        self._sequence = 0
        self._lock = Lock()

    def publish(self, event: dict) -> dict:
        """Write and return an enriched, JSON-compatible telemetry event."""
        with self._lock:
            self._sequence += 1
            payload = dict(event)
            payload.setdefault("event", "transport_tracking")
            payload["sequence"] = self._sequence
            payload["timestamp_unix"] = time.time()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=_json_default,
            )
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=str(self.path.parent),
                text=True,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as temporary:
                    temporary.write(encoded)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_name, self.path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            return payload


_transport_writer = TransportTelemetryWriter()


def publish_transport_tracking(event: dict) -> dict:
    """Publish one transport event through the configured file bridge."""
    try:
        return _transport_writer.publish(event)
    except Exception:
        # Telemetry must never change the robot's execution outcome.
        return dict(event)
