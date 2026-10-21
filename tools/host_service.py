#!/usr/bin/env python3
"""Run this source fork's web host under systemd and load one saved model config.

No installer, downloader or model conversion runs at service startup. Model configs
are prepared separately; the native controller owns all engine processes.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def request(base, path, payload=None, method=None):
    req = urllib.request.Request(base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'}, method=method)
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def notify(message):
    address = os.environ.get('NOTIFY_SOCKET')
    if address:
        if address.startswith('@'):
            address = '\0' + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(address)
            sock.sendall(message.encode())


def main():
    node = os.environ.get('STRATA_NODE') or shutil.which('node')
    if not node:
        raise RuntimeError('Set STRATA_NODE to Node.js 24.12 or newer')
    if not os.environ.get('STRATA_API_KEY'):
        raise RuntimeError('Set the existing provider key in the private service environment file')
    model = os.environ.get('DEFAULT_MODEL', 'ornith-iq3')
    if not (ROOT / f'strata-{model}.json').is_file():
        raise RuntimeError(f'Missing saved model config: strata-{model}.json')
    base = f"http://127.0.0.1:{int(os.environ.get('STRATA_PORT', '10104'))}"
    host = subprocess.Popen([node, 'server/index.mjs', '--dist', 'dist'], cwd=ROOT / 'apps/host')
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        if host.poll() is None:
            host.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for _ in range(120):
            if stopping or host.poll() is not None:
                return host.wait()
            try:
                if request(base, '/api/native?refresh=1').get('available'):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError('The web host did not connect to this source install')
        request(base, '/api/serving', {'apiKey': os.environ['STRATA_API_KEY']}, 'PATCH')
        job = request(base, '/api/native/operations', {'kind': 'load', 'args': {'model': model}})
        notify(f'STATUS=Loading {model}')
        for _ in range(360):
            if stopping or host.poll() is not None:
                return host.wait()
            state = request(base, '/api/native')
            done = next((j for j in state.get('jobs', []) if j['id'] == job['id']), {})
            if done.get('state') == 'error':
                raise RuntimeError(done.get('error', 'Model start failed'))
            if any(i['model'] == model and i['status'] == 'ready' for i in state.get('instances', [])):
                request(base, '/api/setup/complete', {})
                notify(f'READY=1\nSTATUS=Ready: {model}')
                print(f'Strata host ready: {base}, model {model}', flush=True)
                return host.wait()
            time.sleep(1)
        raise TimeoutError('Model did not become ready')
    finally:
        stop()
        try:
            host.wait(timeout=70)
        except subprocess.TimeoutExpired:
            host.kill()
            host.wait()


if __name__ == '__main__':
    raise SystemExit(main())
