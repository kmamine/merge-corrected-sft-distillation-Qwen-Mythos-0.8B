"""Unit tests for distill.merge.merge_state_dicts (exclusion + arithmetic; needs torch)."""
import pytest

torch = pytest.importorskip("torch")

from distill.eval_bench import aggregate, primary_metric
from distill.merge import excluded, merge_state_dicts


class TestExcluded:
    def test_embed_and_head_excluded(self):
        assert excluded("model.embed_tokens.weight")
        assert excluded("lm_head.weight")
        assert not excluded("model.layers.0.self_attn.q_proj.weight")


class TestMergeStateDicts:
    def test_task_arithmetic_on_eligible_tensor(self):
        base = {"w": torch.zeros(4)}
        inst = {"w": torch.ones(4)}
        sft = {"w": torch.full((4,), 2.0)}
        # merged = base + 1*(inst-base) + 1*(sft-base) = 1 + 2 = 3
        merged, n_merged, n_skipped = merge_state_dicts(base, inst, sft, 1.0, 1.0)
        assert n_merged == 1 and n_skipped == 0
        assert torch.allclose(merged["w"], torch.full((4,), 3.0))

    def test_alpha_scales_sft_vector(self):
        base = {"w": torch.zeros(2)}
        inst = {"w": torch.zeros(2)}
        sft = {"w": torch.full((2,), 4.0)}
        merged, _, _ = merge_state_dicts(base, inst, sft, merge_alpha=0.5, chat_alpha=0.0)
        assert torch.allclose(merged["w"], torch.full((2,), 2.0))   # 0.5 * 4

    def test_excluded_key_kept_from_instruct(self):
        base = {"model.embed_tokens.weight": torch.zeros(2)}
        inst = {"model.embed_tokens.weight": torch.ones(2)}
        sft = {"model.embed_tokens.weight": torch.full((2,), 9.0)}
        merged, n_merged, n_skipped = merge_state_dicts(base, inst, sft, 1.0, 1.0)
        assert n_merged == 0 and n_skipped == 1
        assert torch.allclose(merged["model.embed_tokens.weight"], torch.ones(2))

    def test_shape_mismatch_kept_from_instruct(self):
        base = {"w": torch.zeros(2)}
        inst = {"w": torch.ones(3)}        # shape differs
        sft = {"w": torch.zeros(2)}
        merged, n_merged, n_skipped = merge_state_dicts(base, inst, sft)
        assert n_skipped == 1 and torch.allclose(merged["w"], torch.ones(3))


class TestEvalAggregation:
    def test_primary_metric_prefers_exact_match_then_acc_norm_then_acc(self):
        assert primary_metric({"acc,none": 0.4, "exact_match,strict-match": 0.7}) == 0.7
        assert primary_metric({"acc,none": 0.4, "acc_norm,none": 0.5}) == 0.5
        assert primary_metric({"acc,none": 0.42}) == 0.42

    def test_primary_metric_ignores_stderr(self):
        assert primary_metric({"acc,none": 0.3, "acc_stderr,none": 0.01}) == 0.3

    def test_aggregate_mean_ignoring_missing(self):
        assert aggregate({"a": 0.4, "b": 0.6}) == 0.5
        assert aggregate({"a": 0.4, "b": None, "c": float("nan")}) == 0.4
