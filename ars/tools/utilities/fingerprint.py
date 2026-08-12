"""Embedding-model fingerprinting (finding F-10).

A trained checkpoint is only valid together with the embedding model that
produced its semantic vectors. The fingerprint hashes the model directory's
structure (relative paths + sizes) and its small config/text files in full, so
a silently swapped or re-downloaded model is detected without hashing multi-GB
weight files byte-by-byte.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def model_fingerprint(model_path: str | Path) -> str:
    root = Path(model_path)
    if not root.exists():
        raise FileNotFoundError(f'модель эмбеддера не найдена: {root}')
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        h.update(p.relative_to(root).as_posix().encode())
        h.update(str(p.stat().st_size).encode())
        if p.suffix in ('.json', '.txt', '.md') and p.stat().st_size < 1 << 20:
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def verify_fingerprint(model_path: str | Path, expected: str | None) -> None:
    """Raise if the model at `model_path` does not match `expected`.
    `expected` None (legacy artifacts) skips the check."""
    if expected is None:
        return
    actual = model_fingerprint(model_path)
    if actual != expected:
        raise ValueError(
            f'отпечаток модели эмбеддера не совпадает: артефакты обучены с '
            f'{expected}, а по пути {model_path} находится {actual}. '
            f'Подмена эмбеддера молча инвалидирует все чекпоинты (F-10).')
