"""Opt-in, owned llama-server worker for draft architectures Poly does not run.

Strata keeps ownership, authentication and readiness identity. llama.cpp owns
chat templates, multimodal input, sampling and speculative verification. No
inference is sent through Strata's token/embedding pipe on this backend.
"""
from __future__ import annotations

import http.client
import contextlib
import json
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import threading
import time


POST_PATHS = frozenset(("/v1/chat/completions", "/v1/completions", "/v1/messages",
                        "/v1/messages/count_tokens", "/completion", "/tokenize",
                        "/detokenize", "/apply-template"))
GET_PATHS = frozenset(("/v1/models", "/props", "/llama-metrics"))


def backend_args(cfg):
    """Honor the shared context, draft and picture switches on the next load."""
    args = list(cfg["llama_backend"]["args"])
    shared = cfg.get("args", [])
    context = int(shared[shared.index("--max-context") + 1]) if "--max-context" in shared else int(cfg["llama_backend"]["context"])
    args[args.index("-c") + 1] = str(context)
    if cfg.get("mtp") is False:
        for flag in ("-md", "--spec-type", "--spec-draft-n-max", "-ngld"):
            if flag in args:
                at = args.index(flag)
                del args[at:at + 2]
        args += ["--spec-type", "none"]
    if cfg.get("ui_vision_enabled") is False and "--mmproj" in args:
        at = args.index("--mmproj")
        del args[at:at + 2]
        args += ["--no-mmproj-auto"]
    return args, context


class LlamaEngine:
    """A child of the owned Strata server, never an independently running daemon."""
    batch = 0
    terminal_batch_sample = None
    terminal_stage_sample = None

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.args, self.max_context = backend_args(cfg)
        def option(flag, default):
            return self.args[self.args.index(flag) + 1] if flag in self.args else default
        self.info = {"backend": "llama.cpp", "speculative": option("--spec-type", "none"),
                     "kv": option("-ctk", "f16")}
        self.last = {}
        self.log_path = cfg["log"]
        self.supports_images = "--mmproj" in self.args
        self.spawn = (cfg["exe"], cfg.get("args", []))
        self.proc = None
        self.log = None
        self.key_file = None
        self.key = secrets.token_hex(32)
        self.lock = threading.RLock()
        self.connections = set()
        self.start()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        with self.lock:
            if self.alive():
                return
            settings = self.cfg["llama_backend"]
            # The backend stays private; clients only reach the authenticated
            # Strata listener.
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                self.port = sock.getsockname()[1]
            handle = tempfile.NamedTemporaryFile(mode="w", prefix="strata-llama-key-", delete=False)
            self.key_file = Path(handle.name)
            with handle:
                handle.write(self.key)
            self.log = open(self.cfg["log"], "a", encoding="utf-8")
            command = [self.cfg["exe"], *self.args, "--host", "127.0.0.1",
                       "--port", str(self.port), "--api-key-file", str(self.key_file),
                       "--alias", ",".join([self.cfg["model_name"], *self.cfg.get("aliases", [])])]
            try:
                self.proc = subprocess.Popen(command, cwd=self.cfg.get("cwd"), env=self.env,
                                             stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
                deadline = time.monotonic() + settings.get("startup_timeout", 180)
                while self.alive() and time.monotonic() < deadline:
                    conn = self.connection(timeout=2)
                    try:
                        conn.request("GET", "/health", headers=self.headers())
                        response = conn.getresponse()
                        if response.status == 200 and json.loads(response.read()).get("status") == "ok":
                            return
                    except (OSError, ValueError, http.client.HTTPException):
                        pass
                    finally:
                        conn.close()
                    time.sleep(.2)
                raise RuntimeError("llama.cpp did not become ready; inspect " + self.cfg["log"])
            except BaseException:
                self.close()
                raise

    def headers(self):
        return {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}

    def connection(self, timeout=120):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def close(self):
        with self.lock:
            for connection in list(self.connections):
                connection.close()
            if self.alive():
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
            if self.log:
                self.log.close()
                self.log = None
            if self.key_file:
                self.key_file.unlink(missing_ok=True)

    unload = close

    def restart(self):
        self.close()
        self.start()

    def generate(self, *args, **kwargs):
        raise ValueError("This backend accepts HTTP chat requests, not Strata token-pipe generation")


def proxy(handler, svc, path, method):
    """Forward only explicit inference routes, after Strata auth/origin checks.

    Stream the upstream framing as a close-delimited response. read1 preserves
    progressive SSE; a disconnected client closes its upstream request too.
    """
    from serve.server import QueueWaitExpired, QueueCancelled
    engine = svc.engine
    if not isinstance(engine, LlamaEngine):
        return False
    if path not in (POST_PATHS if method == "POST" else GET_PATHS):
        return False
    body = handler._body() if method == "POST" else None
    reservation = svc.reserve_inference()
    if reservation is not True:
        handler._json(503 if reservation is None else 429, {"error": {"message": "worker busy or draining"}})
        return True
    conn = None
    response = None
    sent = False
    admitted = False
    cancel = threading.Event()
    finished = threading.Event()
    handler._watch_client(cancel)

    def cancel_upstream():
        while not finished.wait(.1):
            if cancel.is_set():
                sock = conn.sock if conn is not None else None
                if sock is None and response is not None:
                    sock = getattr(getattr(response.fp, "raw", None), "_sock", None)
                if sock is not None:
                    with contextlib.suppress(OSError):
                        sock.shutdown(socket.SHUT_RDWR)
                return

    threading.Thread(target=cancel_upstream, daemon=True).start()
    with svc.status_lock:
        svc.status["queued"] += 1
    try:
        # Holding the FIFO also makes /unload refuse an in-flight request.
        with svc._inference_fifo(time.monotonic() + 120, cancel):
            with svc.status_lock:
                svc.status["queued"] -= 1
                svc.status["busy"] = True
                admitted = True
            engine.start()
            conn = engine.connection()
            with engine.lock:
                engine.connections.add(conn)
            conn.request(method, "/metrics" if path == "/llama-metrics" else path, body=body, headers=engine.headers())
            response = conn.getresponse()
            handler.send_response(response.status)
            handler.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            handler.end_headers()
            handler.close_connection = True
            sent = True
            while block := response.read1(65536):
                handler.wfile.write(block)
                handler.wfile.flush()
            svc.last_request_at = time.time()
    except (QueueWaitExpired, QueueCancelled):
        if not sent:
            handler._json(503, {"error": {"message": "request cancelled or queue deadline expired"}})
    except (OSError, http.client.HTTPException, RuntimeError):
        if not sent:
            handler._json(502, {"error": {"message": "llama.cpp worker unavailable; inspect its engine log"}})
    finally:
        with svc.status_lock:
            if admitted:
                svc.status["busy"] = False
            else:
                svc.status["queued"] -= 1
        finished.set()
        handler.watch_done.set()
        with svc.status_lock:
            svc.active_cancels.discard(cancel)
        if response is not None:
            response.close()
        if conn:
            conn.close()
            with engine.lock:
                engine.connections.discard(conn)
        svc.release_inference()
    return True
