"""Stage 0 — build the CPT corpus and the QA evaluation set from Aviation_QA.

Writes:
  data/cpt_corpus.jsonl   {"text": <narrative>}             # deduplicated -> CPT input
  data/qa_all.jsonl       {"id","question","answer","context"}
  data/qa_eval.jsonl      a held-out sample of qa_all       # -> eval harness

Run:  python prepare_data.py            (defaults are fine)
      python prepare_data.py --eval_size 500 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random

from aero_cpt.data import load_aviation_qa


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--eval_size", type=int, default=500,
                    help="number of QA pairs held out for evaluation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    print("[prepare] loading Timilehin674/Aviation_QA ...")
    contexts, qa = load_aviation_qa()

    # rough token estimate (~1.3 tokens/word) so you can size the run
    words = sum(len(c.split()) for c in contexts)
    est_tokens = int(words * 1.3)

    corpus_path = os.path.join(args.out_dir, "cpt_corpus.jsonl")
    qa_all_path = os.path.join(args.out_dir, "qa_all.jsonl")
    qa_eval_path = os.path.join(args.out_dir, "qa_eval.jsonl")

    write_jsonl(corpus_path, [{"text": c} for c in contexts])
    write_jsonl(qa_all_path, qa)

    random.shuffle(qa)
    eval_rows = qa[: args.eval_size]
    write_jsonl(qa_eval_path, eval_rows)

    print("[prepare] done.")
    print(f"  unique narratives (CPT corpus) : {len(contexts):,}")
    print(f"  approx CPT tokens              : ~{est_tokens:,}")
    print(f"  QA pairs (all)                 : {len(qa):,}")
    print(f"  QA pairs (eval sample)         : {len(eval_rows):,}")
    print(f"  wrote: {corpus_path}, {qa_all_path}, {qa_eval_path}")


if __name__ == "__main__":
    main()
