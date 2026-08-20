#!/usr/bin/env python3
"""Send a Python source file to the running Isaac Sim VS Code executor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys


def with_environment(source: str, project_root: Path, bridge_root: Path) -> str:
    prelude = (
        "import os\n"
        "import sys\n"
        "import importlib\n"
        "import importlib.machinery\n"
        "from importlib._bootstrap_external import SourceLoader\n"
        f"_aura_source_root = {str(bridge_root)!r}\n"
        "def _aura_source_only_get_code(self, fullname):\n"
        "    _source_filename = self.get_filename(fullname)\n"
        "    if os.path.realpath(_source_filename).startswith(os.path.realpath(_aura_source_root) + os.sep):\n"
        "        return self.source_to_code(self.get_data(_source_filename), _source_filename)\n"
        "    return SourceLoader.get_code(self, fullname)\n"
        "# Idempotent: the fallback above always targets the stdlib implementation.\n"
        "importlib.machinery.SourceFileLoader.get_code = _aura_source_only_get_code\n"
        "_previous_state_module = sys.modules.get('aura_isaac_bridge.core.state')\n"
        "_previous_state = getattr(_previous_state_module, 'state', None)\n"
        "if _previous_state is not None:\n"
        "    for _bridge_name in ('task_bridge', 'camera_bridge'):\n"
        "        _previous_bridge = getattr(_previous_state, _bridge_name, None)\n"
        "        if _previous_bridge is not None and hasattr(_previous_bridge, 'stop'):\n"
        "            try:\n"
        "                _previous_bridge.stop()\n"
        "                print(f'Stopped previous {_bridge_name} before reload')\n"
        "            except Exception as _bridge_stop_error:\n"
        "                print(f'Warning: failed to stop previous {_bridge_name}: {_bridge_stop_error}')\n"
        "import asyncio\n"
        "for _previous_async_task in list(asyncio.all_tasks()):\n"
        "    _previous_coro = _previous_async_task.get_coro()\n"
        "    _previous_owner = getattr(getattr(_previous_coro, 'cr_frame', None), 'f_locals', {}).get('self')\n"
        "    if type(_previous_owner).__name__ in {'IsaacTaskBridge', 'IsaacCameraBridge'}:\n"
        "        _previous_async_task.cancel()\n"
        "        print(f'Cancelled stale AuraVLA bridge task: {type(_previous_owner).__name__}')\n"
        "for _module_name in sorted([_name for _name in list(sys.modules) if _name == 'aura_isaac_bridge' or _name.startswith('aura_isaac_bridge.')], key=lambda _name: _name.count('.'), reverse=True):\n"
        "    _module = sys.modules.pop(_module_name, None)\n"
        "    if _module is not None and '.' in _module_name:\n"
        "        _parent = sys.modules.get(_module_name.rsplit('.', 1)[0])\n"
        "        if _parent is not None and getattr(_parent, _module_name.rsplit('.', 1)[1], None) is _module:\n"
        "            delattr(_parent, _module_name.rsplit('.', 1)[1])\n"
        "importlib.invalidate_caches()\n"
        f"os.environ['AURA_VLA_ROOT'] = {str(project_root)!r}\n"
        f"os.environ['AURA_ISAAC_BRIDGE_ROOT'] = {str(bridge_root)!r}\n"
        f"os.environ['AURA_CAMERA_DIR'] = {os.environ.get('AURA_CAMERA_DIR', '/tmp/aura-vla-camera')!r}\n"
        f"os.environ['AURA_VLA_TASK_DIR'] = {os.environ.get('AURA_VLA_TASK_DIR', '/tmp/aura-vla-control')!r}\n"
        f"os.environ.setdefault('AURA_TRON2_URDF_PATH', {str(project_root / 'aura_description' / 'urdf' / 'tron2_v5_DACH_validing' / 'robot.urdf')!r})\n"
    )
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.lstrip().startswith('from __future__ import'):
            return ''.join(lines[: index + 1]) + prelude + ''.join(lines[index + 1 :])
    return prelude + source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--host', default=os.getenv('AURA_ISAAC_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.getenv('AURA_ISAAC_PORT', '8226')))
    parser.add_argument('--timeout', type=float, default=300.0)
    args = parser.parse_args()
    source_path = args.source.expanduser().resolve()
    if not source_path.is_file():
        print(f'ERROR: Isaac source does not exist: {source_path}', file=sys.stderr)
        return 2
    project_root = Path(os.environ['AURA_VLA_ROOT']).expanduser().resolve()
    bridge_root = project_root / 'aura_hardware' / 'aura_isaac_bridge'
    payload = with_environment(source_path.read_text(encoding='utf-8'), project_root, bridge_root)
    try:
        with socket.create_connection((args.host, args.port), timeout=5.0) as connection:
            connection.settimeout(args.timeout)
            connection.sendall(payload.encode('utf-8'))
            response = connection.recv(16 * 1024 * 1024)
    except OSError as exc:
        print(
            f'ERROR: Isaac Sim VS Code executor is unavailable at {args.host}:{args.port}. '
            'Open Isaac Sim with the VS Code Edition extension enabled.',
            file=sys.stderr,
        )
        print(f'  {exc}', file=sys.stderr)
        return 1
    if not response:
        print('ERROR: Isaac Sim VS Code executor returned an empty response', file=sys.stderr)
        return 1
    try:
        result = json.loads(response.decode('utf-8', errors='replace'))
    except json.JSONDecodeError:
        print('ERROR: Isaac Sim VS Code executor returned invalid JSON', file=sys.stderr)
        print(response[:500].decode('utf-8', errors='replace'), file=sys.stderr)
        return 1
    if result.get('status') != 'ok':
        traceback = '\n'.join(str(line) for line in result.get('traceback') or [])
        print('ERROR: Isaac runtime failed in the existing VS Code Edition process', file=sys.stderr)
        print(traceback or result.get('error') or result.get('evalue') or result, file=sys.stderr)
        return 1
    output = result.get('output')
    if output:
        print(output, end='' if str(output).endswith('\n') else '\n')
    print(f'Isaac runtime loaded in VS Code Edition: {source_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
