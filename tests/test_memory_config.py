"""Tests for MemoryConfig."""

from graphmem import MemoryConfig


def test_from_dict_sets_fields():
    """from_dict should populate dataclass fields from a dict."""
    config = MemoryConfig.from_dict({
        "llm_api_key": "test-key",
        "turso_db_path": "/tmp/x.db",
        "llm_model": "gpt-4o-mini",
    })
    assert config.llm_api_key == "test-key"
    assert config.turso_db_path == "/tmp/x.db"
    assert config.llm_model == "gpt-4o-mini"


def test_from_dict_ignores_unknown_keys():
    """Unknown keys should be silently ignored, not raise TypeError."""
    config = MemoryConfig.from_dict({
        "llm_api_key": "test-key",
        "not_a_real_field": "whatever",
    })
    assert config.llm_api_key == "test-key"


def test_from_dict_empty():
    """Empty dict should produce a default config."""
    config = MemoryConfig.from_dict({})
    assert isinstance(config, MemoryConfig)
