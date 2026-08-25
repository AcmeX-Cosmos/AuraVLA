import json
import tempfile
import unittest
from pathlib import Path

from aura_isaac_bridge.core.telemetry import TransportTelemetryWriter


class TransportTelemetryTest(unittest.TestCase):
    def test_publish_writes_json_atomically_and_increments_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transport_tracking.json"
            writer = TransportTelemetryWriter(path)

            first = writer.publish({"event": "anygrasp_transport_tracking", "error_m": 0.01})
            second = writer.publish({"event": "anygrasp_transport_tracking", "error_m": 0.02})

            self.assertEqual(first["sequence"], 1)
            self.assertEqual(second["sequence"], 2)
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            self.assertEqual(payload["sequence"], 2)
            self.assertEqual(payload["error_m"], 0.02)
            self.assertFalse(list(Path(directory).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
