#!/usr/bin/env python3
"""Register a local GGUF for the optional Poly engine; never download or start a model."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from gguf_reader import GGUFFile
from strata_tokenizer import extract

ARCHITECTURES = {'qwen35moe', 'gemma4', 'glm5next', 'glm5-next', 'deepseek4', 'lfm2moe'}


def plan(model: Path, engine: Path, name: str, config_dir: Path, context: int, gpu: int,
         threads: int, mmproj: Path | None = None, vision_exe: Path | None = None) -> dict:
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,127}', name):
        raise ValueError('name must be a lowercase model id (letters, digits, hyphens and underscores)')
    for path in (model, engine, mmproj, vision_exe):
        if path is not None and not path.is_file():
            raise ValueError(f'file does not exist: {path}')
    if os.name != 'nt' and not os.access(engine, os.X_OK):
        raise ValueError('engine is not executable')
    if engine.stem.lower() not in ('strata-poly', 'strata-glm'):
        raise ValueError('the server identifies the Poly engine by its strata-poly basename')
    if bool(mmproj) != bool(vision_exe):
        raise ValueError('provide both --mmproj and --vision-exe, or neither')
    if not 1 <= context <= 16777216 or gpu < 0 or threads < 1:
        raise ValueError('context and threads must be positive; gpu cannot be negative')
    meta = GGUFFile(model).metadata
    arch = meta.get('general.architecture')
    if arch not in ARCHITECTURES:
        raise ValueError(f'architecture {arch!r} is not one of the imported Poly backends')
    count = int(meta.get('split.count', 1))
    if int(meta.get('split.no', 0)) != 0:
        raise ValueError('provide the first GGUF shard')
    if count > 1:
        match = re.fullmatch(r'(.+)-00001-of-(\d{5})\.gguf', model.name, re.I)
        if not match or int(match[2]) != count:
            raise ValueError('first shard filename does not match split.count')
        for i in range(1, count + 1):
            shard = model.with_name(f'{match[1]}-{i:05d}-of-{count:05d}.gguf')
            if not shard.is_file():
                raise ValueError(f'missing model shard: {shard}')
    dest = config_dir / f'strata-{name}.json'
    pack = config_dir / 'packs' / name
    if dest.exists() or pack.exists():
        raise ValueError('config or pack already exists; choose a new name to keep the existing installation')
    args = ['--model', str(model), '--max-context', str(context), '--threads', str(threads),
            '--expert-profile', str(pack / 'expert-profile.txt'), '--vram-reserve-mib', '1024',
            '--window', '8', '--lookup', '3', '--tail-skip', '0']
    # The Q8 attention cache is supported by Qwen35 (including Ornith and MiniCPM).
    if arch == 'qwen35moe':
        args += ['--kv', 'q8_0']
    cfg = {'exe': str(engine), 'args': args, 'cwd': str(ROOT), 'tokenizer': str(pack / 'tokenizer'),
           'model_name': name, 'display_name': meta.get('general.name', name), 'arch': arch,
           'gpu': gpu, 'host': '127.0.0.1', 'port': 8080, 'parallel': 1,
           'log': str(config_dir / f'{name}.log'),
           'lib_dirs': list(dict.fromkeys(str(p.parent) for p in (engine, vision_exe) if p)),
           'sampling': {'temperature': .6, 'top_p': .95, 'top_k': 20, 'min_p': 0.,
                        'presence_penalty': 0., 'repetition_penalty': 1.}}
    if mmproj:
        cfg['vision'] = {'exe': str(vision_exe), 'model': str(model), 'mmproj': str(mmproj),
                         'gpu': False, 'threads': threads, 'max_tokens': 4096}
    return {'config': cfg, 'path': dest, 'pack': pack, 'architecture': arch, 'shards': count}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'engine', 'config-dir'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--context', type=int, default=32768)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--gpu', type=int, required=True, help='host GPU index (0 inside a container exposing only one card)')
    ap.add_argument('--mmproj', type=Path)
    ap.add_argument('--vision-exe', type=Path)
    ap.add_argument('--api-key-env', help='read the API key from this environment variable; otherwise generate one')
    ap.add_argument('--iq-multi-token', action='store_true', help='opt in to the measured AVX-512 CPU IQ reuse')
    ap.add_argument('--fast-attention', action='store_true', help='opt in to the Poly flash/decode attention kernels')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    try:
        p = plan(a.model.resolve(), a.engine.resolve(), a.name, a.config_dir.resolve(), a.context, a.gpu,
                 a.threads, a.mmproj.resolve() if a.mmproj else None,
                 a.vision_exe.resolve() if a.vision_exe else None)
        env = {}
        if a.iq_multi_token: env['STRATA_POLY_IQ_MT'] = '1'
        if a.fast_attention: env.update(STRATA_FATTN='1', STRATA_FDEC='1')
        if env: p['config']['env'] = env
        if not a.dry_run:
            key = os.environ.get(a.api_key_env) if a.api_key_env else secrets.token_hex(32)
            if not key:
                raise ValueError('the API-key environment variable is empty')
            p['pack'].mkdir(parents=True, mode=0o700)
            extract(a.model.resolve(), p['pack'])
            (p['pack'] / 'expert-profile.txt').touch(mode=0o600)
            cfg = {**p['config'], 'api_key': key}
            with p['path'].open('x', encoding='utf-8') as f:
                os.chmod(p['path'], 0o600)
                json.dump(cfg, f, indent=2, ensure_ascii=False)
                f.write('\n')
        print(json.dumps({'config': str(p['path']), 'architecture': p['architecture'],
                          'shards': p['shards'], 'written': not a.dry_run, 'model_started': False}))
        return 0
    except (OSError, ValueError) as e:
        ap.error(str(e))


if __name__ == '__main__':
    raise SystemExit(main())
