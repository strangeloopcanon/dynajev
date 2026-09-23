# Dynajev: details

## How a request runs

1. **Compile.** Each typed question becomes a field job: which kind of head, which labels.
2. **Bind.** The tokenizer is asked how the labels tokenize. Unique single tokens get a direct slice; phrases get letters; Yes/No and digits are checked to be single tokens. No model call yet.
3. **One forward.** Every field's prompt shares the same prefix (system text plus your state). That prefix is prefilled once. Every field's suffix is then run as one batched forward off a copy of that cache, and the hidden state at each answer boundary is read.
4. **Read.** The compiled rows of the output matrix are dotted with each hidden state. Softmax, sigmoid, or expectation as the type requires. One full-vocabulary matvec per field reports `allowed_mass`.
5. **Decode only what needs it.** `quote` and `open` fields generate; nothing else does.

The answer boundary is an unfinished assistant message: `{"answer": "`. The next token position is the only one read. This is the boundary Simple Jev uses for letters and digits.

### Fitting a head from a few labels

Send `examples` (labeled states for the same question) and two corrections are fit and kept only if they help: a per-class bias plus temperature on the sliced logits (Glance's labeled `fit`), and a ridge probe from the hidden state to the classes (OpenJev's per-task head, without a training run). "Help" means leave-one-out loss goes down and leave-one-out accuracy does not. A temperature cannot change an answer; the bias can. A handful of already-correct examples is not evidence for certainty, so sharpening is capped at 2×. On the models tried so far this rarely changes anything, which the response will tell you.

### Batching many states

`POST /api/decide_batch` takes a list of states and one question set (closed types only) and returns the answers for every state. Two execution modes:

- `shared`: one state at a time, prefilled once, fields branched in one forward. Fewer tokens.
- `dense`: every (state, field) prompt is a row of a padded batch, one forward per chunk. About twice the tokens, but a rectangle.

`auto` picks `dense` on CUDA and `shared` on CPU, because on the measured CPU the dense rectangle processed twice the tokens and lost (61 s vs 46 s for 48 texts × 4 questions); dense is meant for GPUs, where it has not been measured yet.

## Server

`dynajev serve` runs one model on one device, one forward at a time. `/api/ready` returns 503 until the weights are loaded; `/api/health` reports model, dtype, uptime, and cache statistics. Requests queue behind a bounded semaphore and get a 429 when it is full. Prefilled states are kept in a small LRU, so a second question about the same document only pays for its own suffix (a follow-up on a cached ticket answered in 165 ms here; a cold single question takes about 250 ms).

Environment variables: `DYNAJEV_MODEL`, `DYNAJEV_DTYPE` (`bfloat16` on CPU and `float16` on CUDA by default), `DYNAJEV_HOST`, `DYNAJEV_PORT`, `DYNAJEV_MAX_QUEUE`, `DYNAJEV_PREFIX_CACHE`, `DYNAJEV_CORS`.

## Models

The tested trunk is Qwen3.5-2B, loaded as its text-only decoder (the vision tower is not loaded). It is a hybrid: three Gated DeltaNet layers for every full-attention layer. Branching off a shared prefill has to carry recurrent and convolution states as well as key/value caches; the code deep-copies the whole cache, and a unit test checks that a batched branch equals a full forward on that layout. Larger Qwen3.5 checkpoints are the same layout and should work by setting `DYNAJEV_MODEL`.

The readout itself has nothing Qwen-specific in it. Any causal model with a chat template, a decoder at `model.model.layers`, and an `lm_head` will load; anything else is refused with a message pointing at the adapter spot. The heads need Yes, No, the digits, and the letters to be single tokens, which holds for every mainstream tokenizer. What has not been verified on other families is the cache branch. If you try one, run `tests/test_cache.py` against a tiny config of that architecture first.

## Layout

```
src/dynajev/
  compile.py   typed questions and schemas -> field jobs
  bind.py      field jobs + tokenizer -> heads and token rows
  prompts.py   system text, state block, answer boundary
  trunk.py     the frozen model: prefill, branch, dense batch, read rows, decode
  fit.py       affine and ridge corrections gated by leave-one-out
  engine.py    decide / decide_batch
  server.py    FastAPI
  cli.py       serve, ask
web/           Next.js bench
eval/          labeled set and results
scripts/       eval.py, throughput.py, smoke.py
tests/
```
