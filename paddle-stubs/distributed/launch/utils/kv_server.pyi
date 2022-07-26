from __future__ import annotations

import http.server as SimpleHTTPServer
from http.server import HTTPServer
from typing import Any

class KVHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self) -> None: ...
    def do_PUT(self) -> None: ...
    def do_POST(self) -> None: ...
    def do_DELETE(self) -> None: ...
    def output(self, code: Any, value: str = ...) -> None: ...
    def log_message(self, format: Any, *args: Any) -> None: ...

class KVServer(HTTPServer):
    kv_lock: Any = ...
    kv: Any = ...
    port: Any = ...
    stopped: bool = ...
    started: bool = ...
    def __init__(self, port: Any) -> None: ...
    listen_thread: Any = ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

class PKVServer:
    def __init__(self, port: Any) -> None: ...
    proc: Any = ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def started(self): ...
    @property
    def stopped(self): ...
