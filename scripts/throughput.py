"""Many states, one question set: loop vs batch on the same trunk.

Takes the single-question states from eval/items.json (they are short support
notes), asks the same four closed questions of each one, and times three ways:

- loop:  decide() per state, shared prefill + branched fields within a state
- batch: decide_batch() with a padded forward per chunk, several chunk sizes
- write: the same model generating one JSON object per state, greedy, parsed
         (only if --write is passed; it is slow)

    PYTHONPATH=src .venv/bin/python scripts/throughput.py [--states 48] [--write]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dynajev.compile import DecideIn
from dynajev.engine import Dynajev
from dynajev.backends.hf import HFBackend

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer want money back?"},
    "queue": {
        "type": "choice",
        "instructions": "Where should this go?",
        "options": ["billing and payments", "technical support", "account access", "sales inquiry", "shipping"],
    },
    "urgency": {"type": "score", "instructions": "How urgent is this?", "levels": ["low", "medium", "high", "critical"]},
    "flags": {"type": "flags", "instructions": "Which apply?", "options": ["damage", "refund request", "delay", "praise"]},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--states", type=int, default=48)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    items = json.loads(Path("eval/items.json").read_text())
    contexts = []
    for kind in ("boolean", "choice_token", "choice_phrase", "ordinal", "multilabel"):
        contexts.extend(item["context"] for item in items[kind])
    contexts = contexts[: args.states]
    n_questions = len(QUESTIONS)

    trunk = HFBackend.load(args.model, device="cpu")
    engine = Dynajev(trunk)

    # warm up
    engine.decide(DecideIn.model_validate({"context": contexts[0], "questions": QUESTIONS}))

    started = time.perf_counter()
    loop_answers = []
    loop_prefill = 0
    for context in contexts:
        res = engine.decide(DecideIn.model_validate({"context": context, "questions": QUESTIONS}))
        loop_answers.append(res["answers"])
        loop_prefill += res["prefill_tokens"]
    loop_s = time.perf_counter() - started
    decisions = len(contexts) * n_questions
    print(f"loop   {len(contexts)} states x {n_questions} questions: {loop_s:.1f}s  {decisions / loop_s:.1f} decisions/s  {loop_prefill} tokens")

    started = time.perf_counter()
    shared = engine.decide_batch(contexts, QUESTIONS, mode="shared")
    shared_s = time.perf_counter() - started
    print(f"batch shared mode: {shared_s:.1f}s  {decisions / shared_s:.1f} decisions/s  {shared['prefill_tokens']} tokens")

    for chunk in (8, 32):
        started = time.perf_counter()
        batch = engine.decide_batch(contexts, QUESTIONS, chunk_rows=chunk, mode="dense")
        batch_s = time.perf_counter() - started
        agree = sum(
            1
            for a, b in zip(loop_answers, batch["results"])
            for key in a
            if _same(a[key], b["answers"][key])
        )
        print(
            f"batch dense {chunk:>2} rows/forward: {batch_s:.1f}s  {decisions / batch_s:.1f} decisions/s  "
            f"{batch['prefill_tokens']} tokens  agreement with loop {agree}/{decisions}"
        )

    if args.write:
        import torch

        from dynajev.prompts import state_text

        block = (
            "Fill in this JSON object and reply with only the JSON:\n"
            '- "refund": true or false. Does the customer want money back?\n'
            '- "queue": one of ["billing and payments", "technical support", "account access", "sales inquiry", "shipping"].\n'
            '- "urgency": one of ["low", "medium", "high", "critical"].\n'
            '- "flags": a list drawn from ["damage", "refund request", "delay", "praise"].'
        )
        eos = [trunk.tok.eos_token_id]
        started = time.perf_counter()
        tokens = 0
        for context in contexts:
            ids = trunk.encode_prompt(state_text(context) + block, "", "Answer about the state. Reply with only the JSON.")
            with torch.inference_mode():
                out = trunk.model.generate(
                    torch.tensor([ids]), max_new_tokens=80, do_sample=False, eos_token_id=eos, pad_token_id=eos[0]
                )
            tokens += out.shape[1]
        write_s = time.perf_counter() - started
        print(f"write  one JSON per state: {write_s:.1f}s  {decisions / write_s:.1f} decisions/s  {tokens} tokens")


def _same(a: dict, b: dict) -> bool:
    for key in ("noul", "choice", "level", "flags"):
        if key in a:
            return a[key] == b[key] if key != "noul" else abs(a[key] - b[key]) < 0.02
    return a == b


if __name__ == "__main__":
    main()
