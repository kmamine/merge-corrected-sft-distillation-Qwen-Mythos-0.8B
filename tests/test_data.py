"""Unit tests for aero_cpt.data: JSON flattening, token packing, QA prompt building.

A tiny fake tokenizer stands in for a real HF tokenizer so these stay offline.
(Importing aero_cpt.data requires torch, since PackedDataset subclasses Dataset.)
"""
import pytest

torch = pytest.importorskip("torch")

from aero_cpt.data import (PackedDataset, _rows_from_obj, build_qa_inputs,
                      collate_packed, pack_texts)


class FakeTokenizer:
    """One token id per whitespace word (word length, clamped >=1)."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [max(1, len(w)) for w in text.split()]}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        rendered = "\n".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\n<assistant>"
        return rendered


class TestRowsFromObj:
    def test_squad_wrapper(self):
        assert _rows_from_obj({"data": [{"paragraphs": []}]}) == [{"paragraphs": []}]

    def test_bare_list(self):
        assert _rows_from_obj([{"paragraphs": []}]) == [{"paragraphs": []}]

    def test_single_record(self):
        assert _rows_from_obj({"paragraphs": []}) == [{"paragraphs": []}]

    def test_unrecognised_raises(self):
        with pytest.raises(ValueError):
            _rows_from_obj({"nope": 1})


class TestPackTexts:
    def test_blocks_are_exactly_block_size(self):
        tok = FakeTokenizer()
        texts = ["aa bb cc", "dd ee ff", "gg hh ii"]  # 9 word-tokens + 3 EOS = 12 ids
        blocks = pack_texts(texts, tok, block_size=4)
        assert len(blocks) == 3
        assert all(len(b) == 4 for b in blocks)

    def test_drops_incomplete_trailing_block(self):
        tok = FakeTokenizer()
        # 2 word-tokens + 1 EOS = 3 ids; block_size 4 -> 0 full blocks
        blocks = pack_texts(["aa bb"], tok, block_size=4)
        assert blocks == []

    def test_skips_empty_texts(self):
        tok = FakeTokenizer()
        blocks = pack_texts(["", "  ", "aa bb cc dd"], tok, block_size=5)
        # 4 word-tokens + 1 EOS = 5 ids -> exactly one block
        assert len(blocks) == 1


class TestPackedDatasetAndCollate:
    def test_dataset_labels_equal_inputs(self):
        ds = PackedDataset([[1, 2, 3], [4, 5, 6]])
        assert len(ds) == 2
        item = ds[0]
        assert item["input_ids"] == [1, 2, 3]
        assert item["labels"] == [1, 2, 3]

    def test_collate_shapes_and_mask(self):
        batch = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
                 {"input_ids": [4, 5, 6], "labels": [4, 5, 6]}]
        out = collate_packed(batch)
        assert out["input_ids"].shape == (2, 3)
        assert out["labels"].shape == (2, 3)
        assert torch.equal(out["attention_mask"], torch.ones(2, 3, dtype=torch.long))


class TestBuildQaInputs:
    def test_prompt_contains_context_question_and_generation_marker(self):
        tok = FakeTokenizer()
        prompt = build_qa_inputs(tok, context="Engine lost power.", question="What failed?")
        assert "Engine lost power." in prompt
        assert "What failed?" in prompt
        assert prompt.rstrip().endswith("<assistant>")
