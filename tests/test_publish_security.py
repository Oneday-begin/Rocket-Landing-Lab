"""Publication guard regressions; all test values are synthetic."""
import importlib.util
from pathlib import Path

module_path=Path(__file__).resolve().parents[1]/'deployment/check_publish.py'
spec=importlib.util.spec_from_file_location('publish_guard',module_path)
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

def test_public_export_permits_normal_data(tmp_path):
    (tmp_path/'index.html').write_text('<h1>Rocket Landing Lab</h1>')
    (tmp_path/'data').mkdir()
    (tmp_path/'data/flights.json').write_text('[{"score":238.8}]')
    assert guard.inspect(tmp_path)==[]

def test_public_export_blocks_credential_file_and_token(tmp_path):
    (tmp_path/'.env').write_text('LOCAL_SETTING=synthetic')
    (tmp_path/'index.html').write_text('token='+'ghp_'+'aB3cD4eF5gH6iJ7kL8mN9oP0qR1sT2uV3wX4')
    issues=guard.inspect(tmp_path)
    assert any(label=='credential-like filename' for _,label in issues)
    assert any(label=='GitHub token' for _,label in issues)
    assert all('aB3c' not in label for _,label in issues)

def test_public_export_blocks_private_key_and_unknown_path(tmp_path):
    (tmp_path/'notes.txt').write_text('-----BEGIN '+'PRIVATE KEY-----')
    issues=guard.inspect(tmp_path)
    assert any(label=='private key' for _,label in issues)
    assert any(label=='unexpected export path' for _,label in issues)
