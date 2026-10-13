"""Final MedTool-Fine research implementation."""
import hashlib
import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def seal(directory):
    write_json(directory / 'hashes.json', {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'hashes.json'})

def verify(directory):
    for name, expected in read_json(directory / 'hashes.json').items():
        if sha(directory / name) != expected:
            raise ValueError('Index file changed: ' + name)

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'path escapes root: {relative}')
    return path

def now():
    return datetime.now(timezone.utc).isoformat()

