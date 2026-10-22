"""Real per-step simulator telemetry in a compact, queryable SQLite library."""
import argparse
import hashlib
import json
import sqlite3
import zlib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'artifacts' / 'lab'
DB = DATA / 'flights.sqlite'
FIELDS = ['step', 'x', 'y', 'vx', 'vy', 'angle', 'spin', 'left_contact', 'right_contact', 'action', 'reward']


def connect(path=DB):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('''CREATE TABLE IF NOT EXISTS flights (
        id TEXT PRIMARY KEY, model TEXT, scenario TEXT, seed INTEGER,
        model_sha256 TEXT, created TEXT, outcome TEXT, score REAL, steps INTEGER,
        fuel REAL, hover REAL, telemetry BLOB)''')
    return db


def save_flight(db, model, scenario, seed, digest, samples, metrics):
    identity = hashlib.sha256(f'{digest}:{scenario}:{seed}:telemetry-v1'.encode()).hexdigest()[:20]
    packed = zlib.compress(json.dumps(samples, separators=(',', ':'), allow_nan=False).encode())
    with db:
        db.execute('INSERT OR REPLACE INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   (identity, model, scenario, seed, digest, datetime.now(timezone.utc).isoformat(),
                    metrics['outcome'], metrics['base_return'], metrics['steps'],
                    metrics['fuel_proxy'], metrics['hover_fraction'], packed))
    return identity


def collect(model_paths=None, episodes=60, seed=600000, database=DB):
    import torch
    from .models import load_ppo
    from .rewards import make_env
    torch.set_num_threads(1)
    paths = model_paths or sorted((ROOT / 'runs').glob('*/seed_*/final_model.zip'))
    db = connect(database)
    try:
        for path in map(Path, paths):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            label = f'{path.parent.parent.name} / {path.parent.name}'
            policy = load_ppo(path, device='cpu')
            for scenario in ['calm', 'breezy', 'gusty']:
                env = make_env(scenario=scenario)
                try:
                    for i in range(episodes):
                        episode_seed = seed + i
                        identity = hashlib.sha256(f'{digest}:{scenario}:{episode_seed}:telemetry-v1'.encode()).hexdigest()[:20]
                        if db.execute('SELECT 1 FROM flights WHERE id=?', (identity,)).fetchone():
                            continue
                        obs, _ = env.reset(seed=episode_seed)
                        samples = [[0, *np.round(obs, 5).tolist(), -1, 0.0]]
                        for step in range(1, 1001):
                            action = int(policy.predict(obs, deterministic=True)[0])
                            obs, reward, terminal, truncated, info = env.step(action)
                            samples.append([step, *np.round(obs, 5).tolist(), action, round(float(reward), 5)])
                            if terminal or truncated:
                                save_flight(db, label, scenario, episode_seed, digest, samples, info['flight'])
                                break
                        if i % 10 == 0 or i == episodes-1:
                            print(f'Telemetry {label} | {scenario}: {i+1}/{episodes} ({info["flight"]["outcome"]})', flush=True)
                finally:
                    env.close()
    finally:
        db.close()


def flight_rows(database=DB):
    if not Path(database).exists(): return []
    with closing(connect(database)) as db:
        return [dict(r) for r in db.execute('SELECT id,model,scenario,seed,model_sha256,outcome,score,steps,fuel,hover FROM flights ORDER BY model,scenario,seed')]


def read_flight(identity, database=DB):
    with closing(connect(database)) as db:
        row = db.execute('SELECT * FROM flights WHERE id=?', (identity,)).fetchone()
    if row is None: return None
    result = dict(row)
    result['samples'] = json.loads(zlib.decompress(result.pop('telemetry')))
    result['fields'] = FIELDS
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--seed', type=int, default=600000)
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()
    if args.episodes < 1: parser.error('episodes must be positive')
    collect(args.models, args.episodes, args.seed)
