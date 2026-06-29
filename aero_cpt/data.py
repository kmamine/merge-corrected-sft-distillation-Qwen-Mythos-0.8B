"""Data layer for the aviation CPT pipeline.

Handles the one annoyance with `Timilehin674/Aviation_QA`: it ships as SQuAD-nested
JSON and the HF auto-loader can fail the feature cast. `load_aviation_qa` tries the
normal path and falls back to downloading + parsing the raw file(s) by hand.
"""
from __future__ import annotations

import json
import random
from typing import List, Tuple

import torch

AVIATION_QA_REPO = "Timilehin674/Aviation_QA"
QA_SYSTEM_PROMPT = (
    "You are an aviation safety analyst. Answer the question using only the "
    "information in the report. Reply with a short, exact answer and nothing else."
)


# ---------------------------------------------------------------------------
# Loading + flattening
# ---------------------------------------------------------------------------
def _rows_from_obj(obj) -> List[dict]:
    """Normalise a parsed JSON object into a list of records that have 'paragraphs'."""
    if isinstance(obj, dict) and "data" in obj:          # SQuAD wrapper {"data": [...]}
        return list(obj["data"])
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "paragraphs" in obj:
        return [obj]
    raise ValueError("Unrecognised Aviation_QA JSON structure")


def _load_raw_rows() -> List[dict]:
    # Try the clean path first.
    try:
        from datasets import load_dataset

        ds = load_dataset(AVIATION_QA_REPO, split="train")
        rows = [dict(r) for r in ds]
        if rows and "paragraphs" in rows[0]:
            return rows
    except Exception as e:  # noqa: BLE001 - fall back to manual parse
        print(f"[data] load_dataset failed ({type(e).__name__}); parsing raw files.")

    # Fallback: download every .json in the repo and parse (json or jsonl).
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(AVIATION_QA_REPO, repo_type="dataset")
    json_files = [
        f for f in files
        if f.endswith(".json") and "dataset_infos" not in f
    ] or [f for f in files if f.endswith(".jsonl")]
    if not json_files:
        raise RuntimeError(f"No JSON data files found in {AVIATION_QA_REPO}: {files}")

    rows: List[dict] = []
    for fname in json_files:
        path = hf_hub_download(AVIATION_QA_REPO, fname, repo_type="dataset")
        with open(path) as f:
            text = f.read()
        try:
            rows.extend(_rows_from_obj(json.loads(text)))
        except json.JSONDecodeError:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    rows.extend(_rows_from_obj(json.loads(line)))
    return rows


def load_aviation_qa() -> Tuple[List[str], List[dict]]:
    """Return (unique_contexts, qa_pairs).

    unique_contexts -> the CPT corpus (narrative text, deduplicated).
    qa_pairs        -> [{'id', 'question', 'answer', 'context'}] for evaluation.
    """
    rows = _load_raw_rows()
    seen = set()
    contexts: List[str] = []
    qa: List[dict] = []
    for row in rows:
        for para in row.get("paragraphs", []):
            ctx = (para.get("context") or "").strip()
            if not ctx:
                continue
            if ctx not in seen:
                seen.add(ctx)
                contexts.append(ctx)
            for q in para.get("qas", []):
                answers = q.get("answers") or []
                if not answers:
                    continue
                ans = (answers[0].get("text") or "").strip()
                question = (q.get("question") or "").strip()
                if ans and question:
                    qa.append({
                        "id": q.get("id", f"q{len(qa)}"),
                        "question": question,
                        "answer": ans,
                        "context": ctx,
                    })
    return contexts, qa


# ---------------------------------------------------------------------------
# Token packing for CPT
# ---------------------------------------------------------------------------
def pack_texts(texts: List[str], tokenizer, block_size: int) -> List[List[int]]:
    """Tokenize, join docs with EOS, and slice into fixed-length blocks.

    This is the standard causal-LM pretraining packing: no padding, no prompt
    masking — every token is a training target.
    """
    eos = tokenizer.eos_token_id
    stream: List[int] = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        stream.extend(ids)
        if eos is not None:
            stream.append(eos)
    n_blocks = len(stream) // block_size
    return [stream[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]


def build_replay_blocks(tokenizer, block_size: int, n_blocks: int,
                        dataset: str, name, split: str) -> List[List[int]]:
    """Stream a general corpus and pack ~n_blocks blocks of it for replay."""
    if n_blocks <= 0:
        return []
    from datasets import load_dataset

    target_tokens = n_blocks * block_size
    eos = tokenizer.eos_token_id
    stream: List[int] = []
    ds = load_dataset(dataset, name, split=split, streaming=True)
    for ex in ds:
        text = ex.get("text", "")
        if not text or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        stream.extend(ids)
        if eos is not None:
            stream.append(eos)
        if len(stream) >= target_tokens:
            break
    have = len(stream) // block_size
    return [stream[i * block_size:(i + 1) * block_size] for i in range(min(have, n_blocks))]


class PackedDataset(torch.utils.data.Dataset):
    """Wraps a list of equal-length token blocks; labels == input_ids."""

    def __init__(self, blocks: List[List[int]]):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = self.blocks[idx]
        return {"input_ids": ids, "labels": list(ids)}


def collate_packed(batch):
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# QA prompt formatting (used by eval)
# ---------------------------------------------------------------------------
def build_qa_inputs(tokenizer, context: str, question: str, system: str = QA_SYSTEM_PROMPT):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": f"Report:\n{context}\n\nQuestion: {question}",
    })
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
