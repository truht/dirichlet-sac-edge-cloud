"""Socket client implementing /PROTOCOL.md.

Java (EdgeCloudSim) is the TCP server; this process connects once as the
client, before the episode loop starts, and reuses the connection for the
entire training run. Framing: one JSON object per line, newline-terminated,
UTF-8.
"""

import json
import socket


class EdgeCloudSimClient:
    def __init__(self, host="localhost", port=5000):
        self._sock = socket.create_connection((host, port))
        self._reader = self._sock.makefile("r", encoding="utf-8", newline="\n")
        self._writer = self._sock.makefile("w", encoding="utf-8", newline="\n")

    def recv_message(self):
        """Blocks until the next line-delimited JSON message arrives."""
        line = self._reader.readline()
        if not line:
            raise ConnectionError("EdgeCloudSim closed the connection unexpectedly")
        return json.loads(line)

    def send_action(self, episode, time_slot, matrix):
        """matrix: N x (N+1) nested list/array, each row summing to ~1.0."""
        self._send({
            "type": "action",
            "episode": episode,
            "time_slot": time_slot,
            "matrix": matrix,
        })

    def send_shutdown(self):
        """Optional early stop -- see /PROTOCOL.md section 4.4."""
        self._send({"type": "shutdown"})

    def _send(self, message):
        self._writer.write(json.dumps(message))
        self._writer.write("\n")
        self._writer.flush()

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
