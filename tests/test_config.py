"""Unit tests for aero_cpt.config: the dataclass-defaults -> YAML -> overrides chain."""
import textwrap

import pytest

from aero_cpt.config import CPTConfig, load_config, parse_kv


class TestParseKv:
    def test_parses_yaml_typed_scalars(self):
        out = parse_kv(["block_size=512", "use_lora=false", "lr_scheduler=linear"])
        assert out == {"block_size": 512, "use_lora": False, "lr_scheduler": "linear"}

    def test_empty_input_is_empty_dict(self):
        assert parse_kv([]) == {}
        assert parse_kv(None) == {}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError):
            parse_kv(["use_lora"])


class TestLoadConfig:
    def test_defaults_when_no_path(self):
        cfg = load_config(None)
        assert isinstance(cfg, CPTConfig)
        assert cfg.base_model == "Qwen/Qwen2.5-0.5B"
        assert cfg.use_lora is True

    def test_yaml_overrides_defaults(self, tmp_path):
        p = tmp_path / "cpt.yaml"
        p.write_text(textwrap.dedent("""
            use_lora: false
            block_size: 2048
            max_train_minutes: 60
        """))
        cfg = load_config(str(p))
        assert cfg.use_lora is False
        assert cfg.block_size == 2048
        assert cfg.max_train_minutes == 60

    def test_cli_overrides_beat_yaml(self, tmp_path):
        p = tmp_path / "cpt.yaml"
        p.write_text("block_size: 2048\n")
        cfg = load_config(str(p), overrides={"block_size": 256})
        assert cfg.block_size == 256

    def test_unknown_keys_ignored(self, tmp_path):
        p = tmp_path / "cpt.yaml"
        p.write_text("block_size: 128\nnot_a_real_key: 7\n")
        cfg = load_config(str(p))
        assert cfg.block_size == 128
        assert not hasattr(cfg, "not_a_real_key")


class TestTypeCoercion:
    """Overrides are coerced to the dataclass field types.

    Guards the PyYAML gotcha: `yaml.safe_load("2e-5")` returns the *string*
    "2e-5" (its float regex needs a dot), which would otherwise reach AdamW
    as a string. load_config must coerce it to a float per the field annotation.
    """

    def test_scientific_notation_override_becomes_float(self):
        cfg = load_config(None, overrides={"learning_rate": "2e-5"})
        assert isinstance(cfg.learning_rate, float)
        assert cfg.learning_rate == 2e-5

    def test_set_style_scientific_notation_via_parse_kv(self):
        # Mirrors the CLI path:  --set learning_rate=2e-5
        cfg = load_config(None, overrides=parse_kv(["learning_rate=2e-5"]))
        assert isinstance(cfg.learning_rate, float)
        assert cfg.learning_rate == 2e-5

    def test_dotted_float_is_unchanged(self):
        cfg = load_config(None, overrides=parse_kv(["learning_rate=1.0e-4"]))
        assert isinstance(cfg.learning_rate, float)
        assert cfg.learning_rate == 1.0e-4

    def test_bool_and_int_still_typed(self):
        cfg = load_config(None, overrides=parse_kv(["use_lora=false", "block_size=512"]))
        assert cfg.use_lora is False
        assert isinstance(cfg.block_size, int) and cfg.block_size == 512

    def test_int_literal_coerced_to_float_field(self):
        # max_train_minutes is a float field; `--set max_train_minutes=60` -> 60.0
        cfg = load_config(None, overrides=parse_kv(["max_train_minutes=60"]))
        assert isinstance(cfg.max_train_minutes, float)
        assert cfg.max_train_minutes == 60.0

    def test_optional_str_field_keeps_none(self):
        cfg = load_config(None, overrides=parse_kv(["replay_name=null"]))
        assert cfg.replay_name is None

    def test_string_field_stays_string(self):
        cfg = load_config(None, overrides=parse_kv(["lr_scheduler=linear"]))
        assert cfg.lr_scheduler == "linear"
