"""Явный журнал выполнения ноды: print с flush — строки сразу видны в логе запуска SberDS.

Формат: [RCA 12:03:07 +4.2s] этап: сообщение
"""
from __future__ import annotations

import time

_started = time.monotonic()


def start() -> None:
    """Точка отсчёта «+N s» — начало запуска ноды."""
    global _started
    _started = time.monotonic()


def log(stage: str, message: str) -> None:
    print(f'[RCA {time.strftime("%H:%M:%S")} +{time.monotonic() - _started:.1f}s] {stage}: {message}', flush=True)


def preview(text: str, limit: int = 300) -> str:
    """Начало текста одной строкой — чтобы в логе было видно, ЧТО загрузилось."""
    flat = ' '.join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + f'… (+{len(flat) - limit} симв.)'
