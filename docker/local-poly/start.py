"""Seed a separate persistent install, let the original app own its engine, and auto-load one model."""
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import time
import urllib.request

INSTALL = Path('/data/install')
UI = Path('/data/ui/data')
MODELS = {
    'ornith-iq3': 'strata-ornith122b-uncensored-ad-iq3_xxs.json',
    'ornith-iq4': 'strata-ornith122b-uncensored-ad-iq4_xs_s.json',
    'minicpm': 'strata-minicpmv-q4_k_m.json',
    'gemma': 'strata-gemma4-ud-q4_k_m.json',
    'qwen': 'strata-swift-iq3_xxs.json',
}


def private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        os.chmod(temp, 0o600)
        json.dump(value, stream, indent=2)
    temp.replace(path)


def seed():
    # Refresh source on image upgrades; saved model configs, profiles and chats stay on the volume.
    shutil.copytree('/app/source', INSTALL, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name, target in (('.venv', '/opt/strata/.venv'), ('third_party', '/opt/strata/third_party')):
        path = INSTALL / name
        if not path.exists():
            path.symlink_to(target, target_is_directory=True)
    for model, original in MODELS.items():
        dest = INSTALL / f'strata-{model}.json'
        source = Path('/legacy-data/config') / original
        if dest.exists() or not source.exists():
            continue
        cfg = json.loads(source.read_text())
        cfg['exe'] = '/app/engines/strata' if model == 'qwen' else '/app/engines/strata-poly'
        cfg.update(cwd=str(INSTALL), gpu=0, host='127.0.0.1', port=8080, parallel=1,
                   api_key=secrets.token_hex(32), log=f'/data/logs/{model}.log',
                   lib_dirs=['/app/engines', '/usr/local/cuda/lib64'])
        cfg['tokenizer'] = cfg['tokenizer'].replace('/data/', '/legacy-data/', 1)
        args = cfg['args']
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg.startswith('/data/'):
                args[i] = arg.replace('/data/', '/legacy-data/', 1)
        if '--expert-profile' in args:
            index = args.index('--expert-profile') + 1
            profile = Path('/data/profiles') / f'{model}.txt'
            if not profile.exists():
                old = Path(args[index])
                profile.write_bytes(old.read_bytes() if old.exists() else b'')
            args[index] = str(profile)
        if isinstance(cfg.get('vision'), dict):
            cfg['vision'] = {**cfg['vision'], 'exe': '/app/engines/strata-vision', 'gpu': False}
        cfg['env'] = {**cfg.get('env', {}), 'STRATA_FATTN': '1', 'STRATA_FDEC': '1'}
        if model.startswith('ornith'):
            cfg['env']['STRATA_POLY_IQ_MT'] = '1'
        for key in ('before_load', 'expert_profile_save', 'calibration_report'):
            cfg.pop(key, None)
        private_json(dest, cfg)
    if not (UI / 'native-host.json').exists():
        private_json(UI / 'native-host.json', {'repo': str(INSTALL)})


def call(path, payload=None):
    request = urllib.request.Request('http://127.0.0.1:8791' + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    for path in ('/data/logs', '/data/profiles', '/data/workspace', str(UI)):
        Path(path).mkdir(parents=True, exist_ok=True)
    seed()
    host = subprocess.Popen(['node', 'server/index.mjs', '--dist', 'dist'], cwd='/app/host')
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
                if call('/api/native?refresh=1').get('available'):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError('The Strata app did not connect to its local install')
        model = os.environ.get('DEFAULT_MODEL', 'ornith-iq3')
        job = call('/api/native/operations', {'kind': 'load', 'args': {'model': model, 'port': 8080}})
        print('Loading configured model:', model, flush=True)
        for _ in range(360):
            if stopping or host.poll() is not None:
                return host.wait()
            state = call('/api/native')
            done = next((j for j in state.get('jobs', []) if j['id'] == job['id']), {})
            if done.get('state') == 'error':
                for row in call('/api/native/logs').get('logs', [])[-30:]:
                    print(row.get('line', ''), flush=True)
                raise RuntimeError(done.get('error', 'Model start failed'))
            if any(i['status'] == 'ready' for i in state.get('instances', [])):
                call('/api/setup/complete', {})
                print('Strata app and model are ready', flush=True)
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
