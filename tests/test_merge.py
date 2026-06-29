"""Unit tests for distill.merge.merge_state_dicts (exclusion + arithmetic; needs torch)."""
import pytest

torch = pytest.importorskip("torch")

from distill.eval_bench import aggregate, primary_metric, primary_stderr
from distill.merge import excluded, merge_state_dicts


class TestExcluded:
    def test_embed_and_head_excluded(self):
        assert excluded("model.embed_tokens.weight")
        assert excluded("lm_head.weight")
        assert not excluded("model.layers.0.self_attn.q_proj.weight")


class TestMergeMethods:
    def test_linear_interpolates_toward_sft(self):
        inst = {"w": torch.ones(4)}
        sft = {"w": torch.full((4,), 3.0)}
        # linear: instruct + 0.5*(sft - instruct) = 1 + 0.5*2 = 2
        merged, n_merged, n_kept = merge_state_dicts(inst, sft, method="linear", alpha=0.5)
        assert n_merged == 1 and n_kept == 0
        assert torch.allclose(merged["w"], torch.full((4,), 2.0))

    def test_linear_alpha_one_equals_sft(self):
        inst, sft = {"w": torch.zeros(2)}, {"w": torch.full((2,), 4.0)}
        merged, _, _ = merge_state_dicts(inst, sft, method="linear", alpha=1.0)
        assert torch.allclose(merged["w"], torch.full((2,), 4.0))

    def test_ties_trims_small_delta_entries(self):
        inst = {"w": torch.zeros(4)}
        sft = {"w": torch.tensor([10.0, 0.1, 8.0, 0.2])}   # delta == sft
        # density 0.5 keeps the top-2 by |magnitude| (10, 8); alpha=1 -> [10,0,8,0]
        merged, _, _ = merge_state_dicts(inst, sft, method="ties", alpha=1.0, density=0.5)
        assert torch.allclose(merged["w"], torch.tensor([10.0, 0.0, 8.0, 0.0]))

    def test_dare_drop_zero_equals_linear(self):
        inst, sft = {"w": torch.zeros(8)}, {"w": torch.ones(8)}
        merged, _, _ = merge_state_dicts(inst, sft, method="dare_linear", alpha=1.0, drop_p=0.0)
        assert torch.allclose(merged["w"], torch.ones(8))

    def test_dare_is_seeded_deterministic(self):
        inst, sft = {"w": torch.zeros(64)}, {"w": torch.ones(64)}
        a, _, _ = merge_state_dicts(inst, sft, method="dare_linear", alpha=1.0, drop_p=0.5, seed=1)
        b, _, _ = merge_state_dicts(inst, sft, method="dare_linear", alpha=1.0, drop_p=0.5, seed=1)
        assert torch.allclose(a["w"], b["w"])               # same seed -> same drop mask

    def test_slerp_endpoints(self):
        inst, sft = {"w": torch.tensor([1.0, 0.0])}, {"w": torch.tensor([0.0, 2.0])}
        m0, _, _ = merge_state_dicts(inst, sft, method="slerp", slerp_t=0.0)
        m1, _, _ = merge_state_dicts(inst, sft, method="slerp", slerp_t=1.0)
        assert torch.allclose(m0["w"], inst["w"], atol=1e-5)
        assert torch.allclose(m1["w"], sft["w"], atol=1e-5)

    def test_breadcrumbs_keeps_middle_band(self):
        inst = {"w": torch.zeros(4)}
        sft = {"w": torch.tensor([10.0, 5.0, 1.0, 0.1])}      # delta == sft
        # gamma=0.25 drops the top 1 (|10|); density=0.5 keeps top-2 of the rest (5,1) -> [0,5,1,0]
        merged, _, _ = merge_state_dicts(inst, sft, method="breadcrumbs", alpha=1.0,
                                         density=0.5, gamma=0.25)
        assert torch.allclose(merged["w"], torch.tensor([0.0, 5.0, 1.0, 0.0]))

    def test_della_no_drop_equals_linear(self):
        inst, sft = {"w": torch.zeros(8)}, {"w": torch.ones(8)}
        merged, _, _ = merge_state_dicts(inst, sft, method="della", alpha=1.0,
                                         drop_p=0.0, epsilon=0.0)
        assert torch.allclose(merged["w"], torch.ones(8))

    def test_della_seeded_deterministic(self):
        inst, sft = {"w": torch.zeros(64)}, {"w": torch.ones(64)}
        a, _, _ = merge_state_dicts(inst, sft, method="della", alpha=1.0, drop_p=0.5, seed=3)
        b, _, _ = merge_state_dicts(inst, sft, method="della", alpha=1.0, drop_p=0.5, seed=3)
        assert torch.allclose(a["w"], b["w"])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            merge_state_dicts({"w": torch.zeros(2)}, {"w": torch.ones(2)}, method="bogus")

    def test_excluded_key_kept_from_instruct(self):
        inst = {"model.embed_tokens.weight": torch.ones(2)}
        sft = {"model.embed_tokens.weight": torch.full((2,), 9.0)}
        merged, n_merged, n_kept = merge_state_dicts(inst, sft, method="linear", alpha=1.0)
        assert n_merged == 0 and n_kept == 1
        assert torch.allclose(merged["model.embed_tokens.weight"], torch.ones(2))

    def test_shape_mismatch_kept_from_instruct(self):
        inst, sft = {"w": torch.ones(3)}, {"w": torch.zeros(2)}
        merged, n_merged, n_kept = merge_state_dicts(inst, sft, method="linear", alpha=0.5)
        assert n_kept == 1 and torch.allclose(merged["w"], torch.ones(3))


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

    def test_primary_stderr_matches_chosen_metric(self):
        tr = {"exact_match,strict-match": 0.7, "exact_match_stderr,strict-match": 0.012,
              "acc,none": 0.4, "acc_stderr,none": 0.02}
        assert primary_metric(tr) == 0.7              # exact_match preferred
        assert primary_stderr(tr) == 0.012            # its matching stderr

    def test_primary_stderr_none_when_absent(self):
        assert primary_stderr({"acc,none": 0.42}) is None
