"""Synthetic corpus generator: GenConfig.seed drives the corpus (AUDIT_05
minor lead: it was never read, so every seed produced the same corpus)."""
from ars.data.synthesis import DEFAULT_SEED, GenConfig, synthesize


def _corpus(seed: int):
    return synthesize(GenConfig(target_spans=600, seed=seed)).collect()


def test_seed_changes_the_corpus_and_stays_deterministic():
    a, b = _corpus(1), _corpus(2)
    assert not a.equals(b)
    assert a.equals(_corpus(1))


def test_default_seed_is_the_historical_corpus():
    """Salt.of maps every salt onto itself for the default seed, so corpora
    generated before the fix (VALIDATION.md) are reproduced byte for byte."""
    from ars.data.synthesis import Salt
    cfg = GenConfig(seed=DEFAULT_SEED)
    assert Salt.of(cfg, Salt.scenario) == Salt.scenario
    assert _corpus(DEFAULT_SEED).equals(synthesize(GenConfig(target_spans=600)).collect())
