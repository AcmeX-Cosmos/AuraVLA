from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import sys
import time

# The launcher can be imported from a colcon ``install`` tree while Isaac must
# hot-reload sources from the active workspace. Resolve the workspace once and
# use it for every injected runtime file.
def _is_aura_workspace(path: Path) -> bool:
    return (
        (path / "aura_bringup" / "config" / "config.yaml").is_file()
        and (path / "aura_hardware" / "aura_isaac_bridge" / "robot" / "robot.py").is_file()
    )


def _find_aura_workspace() -> Path:
    configured_root = os.environ.get("AURA_VLA_ROOT")
    if configured_root:
        candidate = Path(configured_root).expanduser().resolve()
        if _is_aura_workspace(candidate):
            return candidate

    # This covers direct source execution. The environment variable is the
    # required mechanism when this module originates from ``install``.
    for origin in (Path(__file__).resolve(), Path.cwd().resolve()):
        for candidate in (origin, *origin.parents):
            if _is_aura_workspace(candidate):
                return candidate
    raise RuntimeError(
        "Unable to locate the AuraVLA workspace. Set AURA_VLA_ROOT to the "
        "project directory before starting the NVIDIA agent."
    )


# Add execution path for FileTaskClient import.
_aura_root = _find_aura_workspace()
_execution_path = _aura_root / "aura_execution" / "aura_execution"
for _path in (
    _execution_path,
    _aura_root / "aura_orchestration",
    _aura_root / "aura_planning",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from aura_execution.task_bridge import FileTaskClient
except ImportError:
    from task_bridge import FileTaskClient


class IsaacRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class IsaacRuntimeConfig:
    host: str = "127.0.0.1"
    port: int = 8226
    connect_timeout_sec: float = 5.0
    execution_timeout_sec: float = 300.0
    ready_timeout_sec: float = 45.0
    poll_interval_sec: float = 0.1


class IsaacRuntimeLauncher:
    """Restores the AuraVLA Isaac bridges through the VS Code executor."""

    def __init__(
        self,
        task_client: FileTaskClient | None,
        *,
        config: IsaacRuntimeConfig | None = None,
        entry_path: str | Path | None = None,
        reload_entry_path: str | Path | None = None,
        camera_entry_path: str | Path | None = None,
        camera_directory: str | Path = "/tmp/aura-vla-camera",
        source_sender: Callable[[str], str] | None = None,
    ) -> None:
        self.task_client = task_client
        self.config = config or IsaacRuntimeConfig()
        self.project_root = _aura_root
        self.camera_directory = Path(camera_directory).expanduser().resolve()
        source_bridge_root = (
            self.project_root / "aura_hardware" / "aura_isaac_bridge"
        )
        self.entry_path = Path(
            entry_path or source_bridge_root / "robot" / "robot.py"
        ).expanduser().resolve()
        self.reload_entry_path = Path(
            reload_entry_path or source_bridge_root / "robot" / "robot.py"
        ).expanduser().resolve()
        self.camera_entry_path = Path(
            camera_entry_path or source_bridge_root / "start_camera_bridge.py"
        ).expanduser().resolve()
        self._source_sender = source_sender or self._send_source

    def ensure_ready(self) -> bool:
        if self.task_client is not None and self.task_client.is_ready():
            return False
        if not self.entry_path.is_file():
            raise IsaacRuntimeError(
                f"Isaac runtime entry does not exist: {self.entry_path}"
            )

        source = self._with_camera_environment(
            self.entry_path.read_text(encoding="utf-8")
        )
        try:
            raw_response = self._source_sender(source)
        except OSError as exc:
            raise IsaacRuntimeError(
                "Isaac VSCode executor is unavailable at "
                f"{self.config.host}:{self.config.port}. Start Isaac Sim with the "
                "VS Code Edition extension enabled."
            ) from exc

        self._validate_response(raw_response)
        if self.task_client is None:
            return True
        deadline = time.monotonic() + self.config.ready_timeout_sec
        while time.monotonic() < deadline:
            if self.task_client.is_ready():
                return True
            time.sleep(self.config.poll_interval_sec)
        raise IsaacRuntimeError(
            f"Isaac runtime loaded but task bridge did not become ready: "
            f"{self.task_client.paths.status}"
        )

    def restore_camera(self) -> None:
        if not self.camera_entry_path.is_file():
            raise IsaacRuntimeError(
                f"Isaac camera entry does not exist: {self.camera_entry_path}"
            )
        source = self._with_camera_environment(
            self.camera_entry_path.read_text(encoding="utf-8")
        )
        # The VS Code executor evaluates source with its own globals, so a
        # script's ``if __name__ == '__main__'`` block is not reliable. The
        # current launcher is exec-safe already; this explicit fallback also
        # keeps older camera scripts from becoming a silent no-op.
        source += (
            "\n\n"
            "if 'start_camera_bridge' in globals() and "
            "globals().get('camera_bridge') is None:\n"
            "    camera_bridge = start_camera_bridge()\n"
        )
        try:
            raw_response = self._source_sender(
                source
            )
        except OSError as exc:
            raise IsaacRuntimeError(
                "Isaac VSCode executor is unavailable at "
                f"{self.config.host}:{self.config.port}. Start Isaac Sim with the "
                "VS Code Edition extension enabled."
            ) from exc
        self._validate_response(raw_response)

    def reload(self) -> None:
        if not self.reload_entry_path.is_file():
            raise IsaacRuntimeError(
                f"Isaac runtime reload entry does not exist: {self.reload_entry_path}"
            )
        try:
            raw_response = self._source_sender(
                self._with_camera_environment(
                    self.reload_entry_path.read_text(encoding="utf-8")
                )
            )
        except OSError as exc:
            raise IsaacRuntimeError(
                "Isaac VSCode executor is unavailable at "
                f"{self.config.host}:{self.config.port}. Start Isaac Sim with the "
                "VS Code Edition extension enabled."
            ) from exc
        self._validate_response(raw_response)

    def _send_source(self, source: str) -> str:
        with socket.create_connection(
            (self.config.host, self.config.port),
            timeout=self.config.connect_timeout_sec,
        ) as connection:
            connection.settimeout(self.config.execution_timeout_sec)
            connection.sendall(source.encode("utf-8"))
            response = connection.recv(16 * 1024 * 1024)
        if not response:
            raise IsaacRuntimeError("Isaac VSCode executor returned an empty response")
        return response.decode("utf-8", errors="replace")

    def _with_camera_environment(self, source: str) -> str:
        camera_directory = repr(str(self.camera_directory))
        project_root = str(self.project_root)
        bridge_root = str(self.project_root / "aura_hardware" / "aura_isaac_bridge")
        prelude = (
            "import os\n"
            "import sys\n"
            "import importlib\n"
            "_previous_state_module = sys.modules.get('aura_isaac_bridge.core.state')\n"
            "_previous_state = getattr(_previous_state_module, 'state', None)\n"
            "if _previous_state is not None:\n"
            "    for _model_attr in ('_sam_model', '_anygrasp_model', '_graspnet_net'):\n"
            "        _previous_model = getattr(_previous_state, _model_attr, None)\n"
            "        if _previous_model is not None and hasattr(_previous_model, 'to'):\n"
            "            try:\n"
            "                _previous_model.to('cpu')\n"
            "            except Exception:\n"
            "                pass\n"
            "        setattr(_previous_state, _model_attr, None)\n"
            "    _previous_state._graspnet_demo = None\n"
            "for _module_name in sorted([_name for _name in list(sys.modules) if _name == 'aura_graspnet_demo' or _name == 'aura_isaac_bridge' or _name.startswith('aura_isaac_bridge.') or _name == 'pointnet2' or _name.startswith('pointnet2.') or _name == 'gsnet' or _name.startswith('gsnet.')], key=lambda _name: _name.count('.'), reverse=True):\n"
            "    _module = sys.modules.pop(_module_name, None)\n"
            "    if _module is not None and '.' in _module_name:\n"
            "        _parent = sys.modules.get(_module_name.rsplit('.', 1)[0])\n"
            "        if _parent is not None and getattr(_parent, _module_name.rsplit('.', 1)[1], None) is _module:\n"
            "            delattr(_parent, _module_name.rsplit('.', 1)[1])\n"
            "import gc\n"
            "gc.collect()\n"
            "importlib.invalidate_caches()\n"
            f"os.environ['AURA_VLA_ROOT'] = {project_root!r}\n"
            f"os.environ['AURA_ISAAC_BRIDGE_ROOT'] = {bridge_root!r}\n"
            f"os.environ['AURA_CAMERA_DIR'] = {camera_directory}\n"
            f"os.environ.setdefault('AURA_TRON2_URDF_PATH', {str(self.project_root / 'aura_description' / 'urdf' / 'tron2_v5_DACH_validing' / 'robot.urdf')!r})\n"
        )
        lines = source.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.lstrip().startswith("from __future__ import"):
                insert_at = index + 1
                return "".join(lines[:insert_at]) + prelude + "".join(
                    lines[insert_at:]
                )
        return prelude + source

    @staticmethod
    def _validate_response(raw_response: str) -> None:
        try:
            response = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise IsaacRuntimeError(
                f"Isaac VSCode executor returned invalid JSON: {raw_response[:300]}"
            ) from exc
        if response.get("status") != "ok":
            traceback_lines = response.get("traceback") or []
            traceback_text = "\n".join(str(line) for line in traceback_lines).strip()
            message = (
                traceback_text
                or response.get("error")
                or response.get("evalue")
                or response.get("output")
                or raw_response
            )
            raise IsaacRuntimeError(f"Isaac runtime failed to load: {message}")
