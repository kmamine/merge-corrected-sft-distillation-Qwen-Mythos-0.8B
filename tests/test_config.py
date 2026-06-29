"""Unit tests for distill.config: dataclass-defaults -> YAML -> overrides + coercion."""
import textwrap

import pytest

from distill.config import SFTMergeConfig, load_config, parse_kv


class TestParseKv:
    def test_parses_yaml_typed_scalars(self):
        out = parse_kv(["per_device_batch_size=8", "use_lora=false", "lr_scheduler=linear"])
        assert out == {"per_device_batch_size": 8, "use_lora": False, "lr_scheduler": "linear"}

    def test_empty_and_none(self):
        assert parse_kv([]) == {}
        assert parse_kv(None) == {}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError):
            parse_kv(["use_lora"])


class TestLoadConfig:
    def test_defaults(self):
        cfg = load_config(None)
        assert isinstance(cfg, SFTMergeConfig)
        assert cfg.base_model == "Qwen/Qwen3.5-0.8B-Base"
        assert cfg.instruct_model == "Qwen/Qwen3.5-0.8B"
        assert cfg.dataset == "WithinUsAI/claude_mythos_distilled_25k"
        assert cfg.num_epochs == 3

    def test_yaml_then_cli_override(self, tmp_path):
        p = tmp_path / "sft.yaml"
        p.write_text(textwrap.dedent("""
            num_epochs: 2
            merge_alpha: 0.5
        """))
        cfg = load_config(str(p), overrides={"num_epochs": 5})
        assert cfg.num_epochs == 5          # CLI beats YAML
        assert cfg.merge_alpha == 0.5

    def test_unknown_keys_ignored(self, tmp_path):
        p = tmp_path / "sft.yaml"
        p.write_text("num_epochs: 1\nnot_real: 7\n")
        cfg = load_config(str(p))
        assert cfg.num_epochs == 1
        assert not hasattr(cfg, "not_real")


class TestTypeCoercion:
    def test_scientific_notation_to_float(self):
        cfg = load_config(None, overrides=parse_kv(["learning_rate=2e-5"]))
        assert isinstance(cfg.learning_rate, float) and cfg.learning_rate == 2e-5

    def test_int_literal_to_float_field(self):
        cfg = load_config(None, overrides=parse_kv(["merge_alpha=1"]))
        assert isinstance(cfg.merge_alpha, float) and cfg.merge_alpha == 1.0

    def test_bool_and_int(self):
        cfg = load_config(None, overrides=parse_kv(["use_lora=true", "num_epochs=4"]))
        assert cfg.use_lora is True
        assert isinstance(cfg.num_epochs, int) and cfg.num_epochs == 4

    def test_optional_str_keeps_none(self):
        cfg = load_config(None, overrides=parse_kv(["mlflow_experiment=null"]))
        assert cfg.mlflow_experiment is None

    def test_string_field_stays_string(self):
        cfg = load_config(None, overrides=parse_kv(["eval_tasks=gsm8k,mmlu"]))
        assert cfg.eval_tasks == "gsm8k,mmlu"
