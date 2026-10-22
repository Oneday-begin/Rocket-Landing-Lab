import copy
import csv
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rocket_landing import telemetry
from rocket_landing.predictor import features, predict_flight, HORIZON
from rocket_landing.broader_training import RandomWeather
from rocket_landing.rewards import make_env
from rocket_landing.webapp import app


def test_telemetry_roundtrip_and_identity(tmp_path):
    path = tmp_path / 'flights.sqlite'
    db = telemetry.connect(path)
    samples = [[0, 0, 1, 0, -0.1, 0, 0, 0, 0, -1, 0],
               [1, 0, .99, 0, -.08, 0, 0, 0, 0, 2, 2.7]]
    metrics = dict(outcome='timeout', base_return=2.7, steps=1, fuel_proxy=.3, hover_fraction=0)
    identity = telemetry.save_flight(db, 'test', 'calm', 1, 'hash', samples, metrics)
    assert identity == telemetry.save_flight(db, 'test', 'calm', 1, 'hash', samples, metrics)
    db.close()
    assert len(telemetry.flight_rows(path)) == 1
    result = telemetry.read_flight(identity, path)
    assert result['samples'] == samples
    assert result['score'] == sum(s[-1] for s in samples)
    assert result['fields'][9] == 'action'


def test_predictor_never_reads_future():
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(400, 11)).tolist()
    for s in samples: s[9] = 2
    expected = features(samples)
    changed = copy.deepcopy(samples)
    changed[HORIZON+1:] = [[9999] * 11 for _ in changed[HORIZON+1:]]
    np.testing.assert_array_equal(features(changed), expected)
    assert expected.shape == (32,)
    assert expected[-2] == 1 and expected[-1] == 0
    with pytest.raises(ValueError): features(samples[:150])


def test_early_ending_is_not_forecast():
    assert not predict_flight({'steps':150})['available']


def test_saved_predictor_has_disjoint_seed_partitions():
    report = json.loads((telemetry.DATA / 'predictor-report.json').read_text())
    groups = [set(v['seeds']) for v in report['splits'].values()]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i+1:])
    assert sum(v['flights'] for v in report['splits'].values()) == report['eligible_flights']
    assert sum(map(sum, report['test']['confusion_matrix'])) == report['splits']['test']['flights']


def test_random_weather_is_seeded_and_within_range():
    env = RandomWeather(make_env(scenario='gusty'))
    try:
        first, _ = env.reset(seed=12)
        weather = env.unwrapped.wind_power, env.unwrapped.turbulence_power
        second, _ = env.reset(seed=12)
        np.testing.assert_array_equal(first, second)
        assert weather == (env.unwrapped.wind_power, env.unwrapped.turbulence_power)
        assert 0 <= weather[0] <= 18 and 0 <= weather[1] <= 1.8
        env.reset()
        assert weather != (env.unwrapped.wind_power, env.unwrapped.turbulence_power)
    finally: env.close()


def test_api_filters_statistics_and_csv():
    with TestClient(app) as client:
        rows = client.get('/api/flights?scenario=gusty').json()
        stats = client.get('/api/summary?scenario=gusty').json()
        assert rows and all(r['scenario'] == 'gusty' for r in rows)
        assert stats['count'] == len(rows)
        assert stats['landed'] == sum(r['outcome']=='landed' for r in rows)
        exported = list(csv.DictReader(io.StringIO(client.get('/api/export?scenario=gusty').text)))
        assert len(exported) == len(rows)
        assert set(r['id'] for r in exported) == set(r['id'] for r in rows)
        assert client.get('/api/flights/'+rows[0]['id']).json()['samples']
        assert client.get('/api/summary?model=missing').json()['count'] == 0


def test_api_rejects_bad_requests_and_serves_entry():
    with TestClient(app) as client:
        assert 'A small rocket.' in client.get('/').text
        assert client.get('/assets/app.js').status_code == 200
        assert client.get('/api/flights/not-a-flight').status_code == 404
        assert client.post('/api/ask',json={'question':'x'}).status_code == 422
        assert client.post('/api/jobs',json={'kind':'invalid'}).status_code == 400
        assert client.post('/api/jobs',json={'kind':'train','steps':-1}).status_code == 422
        assert client.post('/api/jobs',json={'kind':'predictor'},headers={'origin':'https://example.com'}).status_code == 403


def test_assistant_queries_real_data_and_retrieves_evidence(monkeypatch):
    from rocket_landing import assistant
    rows = telemetry.flight_rows()
    answer = assistant.data_answer('How many flights are there in gusty weather?')
    assert str(sum(r['scenario']=='gusty' for r in rows)) in answer['answer']
    assert answer['mode'] == 'computed data'
    sources = assistant.retrieve('Why did the hover policy fail?')
    assert sources and any('hover' in s['text'].lower() for s in sources)
    assert assistant.retrieve('zxqvbnm987654321') == []
    def offline(*args, **kwargs): raise OSError('offline')
    monkeypatch.setattr(assistant.request,'urlopen',offline)
    assert assistant.answer('How does PPO work?')['mode'] == 'retrieved excerpts'
