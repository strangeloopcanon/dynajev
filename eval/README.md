# Eval

`items.json` holds the labeled set. `../scripts/eval.py` runs every item as a
read (compiled head) and a write (greedy chat completion, parsed) on the same
trunk and writes `RESULTS.md` and `results.json` here.

    PYTHONPATH=src .venv/bin/python scripts/eval.py

Runs take a few minutes on CPU. Stop the API server first if memory is tight;
both hold a copy of the weights.
