# Dynajev: details

## How a request runs

1. **Compile** (`compile.py`). Each typed question becomes a field job: kind, labels, dependencies. The type is declared. A legacy single `question` may leave `type` out only when the shape decides it (`levels` means score, `options` with `exclusive: false` means flags, `options` means choice); otherwise the request is refused. The wording of a question is never parsed.
2. **Bind** (`bind.py`). The tokenizer is asked how the labels tokenize, and each job becomes a `FieldPlan`: its branches (prompts), its reads (which rows at which position), a combiner, or a decode. No model call yet.
3. **Plan** (`plan.py`). The field plans, grouped into stages by `depends_on` / `include_answers`, are the request's IR.
4. **Execute** (`executor.py`). Each stage's branches go to the trie reader, which returns one hidden state per branch; reads and combiners turn them into answers. Fields that need text decode after that.

The answer boundary is an unfinished assistant message, `{"answer": "`. The next position is the only one read.

Every response carries `plan`, a compact account of what was built:

```json
{
  "summary": "7 branches in 1 stage; 1 shared segment; 2 forwards; 328 of 640 prompt tokens processed",
  "stages": [["refund", "queue", "stars", "flags"]],
  "fields": [
    {"id": "refund", "type": "noul", "head": "binary_margin", "branches": 1, "tokens": [80],
     "read": "4 unembedding rows", "combine": "sigmoid(Yes - No)", "depth": "24/24 layers"},
    {"id": "flags", "type": "flags", "head": "multilabel_margin", "branches": 4, "tokens": [87, 88, 87, 87],
     "read": "4 unembedding rows at 4 positions", "combine": "4 independent sigmoids", "depth": "24/24 layers"}
  ],
  "reads": [{"branches": 7, "naive_tokens": 640, "processed_tokens": 328, "forwards": 2, "shared_prefix_tokens": 56}]
}
```

(Abridged.) `fields`, `answers`, `prefill_tokens`, `generated_tokens`, `notes` and `fit` are unchanged from earlier versions.

## IR

| Node | Fields | Meaning |
| --- | --- | --- |
| `Branch` | id, field, block, system, assistant prefix, label, tokens, depth, keep | One prompt that ends at an answer boundary. A field with several branches (flags, per-option margins, criteria) has one per label. |
| `Read` | branch, op (`rows`, `prototype`, `hidden`), token groups, layer | What to take from that branch's hidden state: log-sum-exp scores of token groups, prototype scores, or the raw state for a fitted probe. |
| `Combine` | op, labels, correction | How a field's reads become an answer: `sigmoid`, `softmax`, `expectation`, `flags`, `margin_softmax`, `criteria`. `correction` is a fitted or stored head (bias and temperature, or a ridge probe). |
| `Decode` | branch, mode, extractive | A short greedy decode (`quote`, schema strings) or a chat completion (`open`). |

## Trie prefill

A request's branch token sequences form a trie (`trie.py`). A segment shared by several branches can be prefilled once and forked (the cache is deep-copied, which on Qwen3.5 carries the DeltaNet conv and recurrent states as well as key/values). Siblings below a node run as one right-padded batch off its cache; causality means the padding cannot reach a row's last real token, so the result is exact. `tests/test_cache.py` checks every branch's hidden state against a full forward of that branch on a tiny Qwen3.5 (atol 1e-4), for several trie shapes and depths.

Whether a shared segment gets its own forward is a cost decision. On the measured CPU a forward costs about 100 ms before any tokens (streaming the weights) plus about 1.5 ms per token. Splitting a segment of length L shared by n rows saves (n − 1)·L tokens and costs one extra forward, so it is split only when that saving is at least `DYNAJEV_FORWARD_OVERHEAD` tokens (default 64; 0 splits everything). On the README's four-question request (measured on the previous, slightly longer templates): 318 of 696 tokens in 2 forwards, 1.10 s; splitting everything, 267 tokens in 4 forwards, 1.17 s; no sharing, 696 tokens, 1.62 s. On a GPU the fixed cost is smaller, so a lower setting is likely better.

Templates put the state first and each field's label last, so branches of a field share the question as well as the state.

`include_answers` continues a parent's conversation. The parent branch is marked `keep`, its cache is kept after the read, and the child's prompt is the parent's tokens, the parent's answer, and a new user turn. The child's trie then starts at the parent's full prompt. The continuation is built by appending tokens rather than re-rendering the chat template, because Qwen3.5's template drops the empty think block from earlier assistant turns and re-rendering would not reproduce the parent's prefix.

## Early exit

Given labeled `examples`, each closed single-branch field (noul, choice, score) is fit as before: a bias and temperature on the sliced logits, or a ridge probe on the final hidden state, kept only if leave-one-out loss improves and accuracy does not drop. The example forwards also tap candidate layers (every fourth layer and the last; every layer below 8). A calibrated ridge probe is fit at each candidate layer, shallowest first, and the first one whose leave-one-out accuracy is at least the full-depth head's and whose leave-one-out log loss is at most 0.1 nats worse is chosen. The field then runs only that many layers: the decoder's layer list is sliced for the forward and the final norm is skipped. The response's `fit.layer_scan` shows each candidate's accuracy and loss, and `fit.full_depth_loo` the baseline.

Truncated states are tested against `output_hidden_states` of an untruncated forward. Taps are taken with forward hooks, not `output_hidden_states`, because transformers installs its hidden-state hooks once on the layers present at the first call.

On the tone set, fitted on 8 states, a probe after layer 12 of 24 passed (layers 4 and 8 did not). Applied as a stored head to the other 8 states it answered in a median 108 ms against 238 ms for the zero-shot full-depth read, and got 7 of 8 right against 8 of 8. A request carrying the 8 examples takes about 1.8 s; the example forwards run as one batch. With few examples the probe is fitted on very little data; the gate protects against a worse leave-one-out score, not against a small or unrepresentative example set.

## Stored heads

A fitted head is keyed by a task signature: a hash of the template version, model id, type, instructions (whitespace and case collapsed), labels in order, strategy, and criteria. `save_heads: true` on `/api/decide` stores it, or `POST /api/heads/fit` with `{"questions": {...}, "examples": [...]}`. Later requests whose field has the same signature use it automatically (turn off with `use_heads: false`); `notes` and `plan.fields[].source` say so, and a head with an exit layer runs truncated. `GET /api/heads` lists them. The store is an in-memory LRU (256 heads), backed by `<signature>.json` plus `<signature>.npz` files when `DYNAJEV_HEAD_STORE` names a directory. Changing a template bumps `TEMPLATE_VERSION`, which invalidates every stored head.

## Dependent questions

```json
"remedy": {"type": "choice", "instructions": "What should we send?", "options": ["a replacement", "a refund"],
           "depends_on": {"question": "damaged", "when": true}, "include_answers": ["damaged"]}
```

`depends_on` runs a question only if its parent's answer matches `when` (a value or a list; for flags, any listed flag being set). A skipped question answers `{"type": "choice", "skipped": true}`. `include_answers` shows the question earlier answers: the first listed parent that is a single-branch closed question is continued as a conversation (above); anything else is written into the prompt as `Earlier answers:` lines. Questions are grouped into stages, each needing only earlier stages; unknown ids, self-references, `when` values that are not a possible answer, and cycles are compile errors. Each stage is one trie read, so a request with dependencies costs one extra read per stage.

## Criteria

A plain `noul` is one comparative margin: Yes rows minus No rows at one boundary. With `criteria: {"true": "...", "false": "..."}`, each description is judged on its own branch, as "does this description fit the state?", without seeing the other. The probability is the sigmoid of the true margin minus the false margin, and the answer also reports each judgment and `ambiguous` when both fit or neither does. It costs one branch per criterion and cannot use a stored head.

## Backends

`backends/base.py` defines the protocol the executor uses: `encode_prompt`, `continue_prompt`, `prefill(tokens, cache, layers)`, `prefill_rows(cache, rows, layers)`, `fork`, `read_dense`, `class_logits`, `prototype_logits`, `as_vector`, `generate`, plus `num_layers`, `hidden_size`, `vocab_size`. `backends/hf.py` is the transformers implementation and the only one; `DYNAJEV_BACKEND` selects it.

There is no vLLM or SGLang backend. What one would need:

- The final hidden state at the answer position, or at least logits for chosen token ids there. `prompt_logprobs` with a fixed top-k is not enough: the allowed rows are often outside the top-k, and `allowed_mass` needs the full distribution. vLLM's pooling runner can return hidden states but not in the same pass as generation.
- Automatic prefix caching gives the trie's saving for free when branches are sent together, but not the cost-model choice of which segments to split, and not cache forks under the caller's control.
- Early exit needs a partial forward to a layer, which neither exposes.
- Fitted probes and the hidden-state reads need the raw residual stream at a layer, which neither exposes as a stable API.

A backend that only returns logits for chosen tokens could serve the zero-shot heads (noul, choice, score, flags, criteria) and affine corrections, but not ridge probes or early exit.

## Batching many states

`POST /api/decide_batch` takes a list of states and one question set (closed types only). Two modes:

- `shared`: one state at a time through the trie. Fewer tokens.
- `dense`: every (state, field) prompt is a row of a padded batch, one forward per chunk. About twice the tokens, but a rectangle.

`auto` picks `dense` on CUDA and `shared` on CPU. Question sets with dependencies run `shared`; asking for `dense` with them is an error.

## Server

`dynajev serve` runs one model on one device, one forward at a time. `/api/ready` returns 503 until the weights are loaded; `/api/health` reports model, dtype, layers, uptime, and cache statistics. Requests queue behind a bounded semaphore and get a 429 when it is full. Prefilled states are kept in a small LRU, so a second question about the same state pays only for its own suffix.

Environment variables: `DYNAJEV_MODEL`, `DYNAJEV_DTYPE` (`bfloat16` on CPU and `float16` on CUDA by default), `DYNAJEV_HOST`, `DYNAJEV_PORT`, `DYNAJEV_MAX_QUEUE`, `DYNAJEV_PREFIX_CACHE`, `DYNAJEV_FORWARD_OVERHEAD`, `DYNAJEV_HEAD_STORE`, `DYNAJEV_BACKEND`, `DYNAJEV_CORS`.

## Models

The tested trunk is Qwen3.5-2B, loaded as its text-only decoder. It is a hybrid: three Gated DeltaNet layers for every full-attention layer, so branching has to carry recurrent and convolution states as well as key/value caches. Larger Qwen3.5 checkpoints have the same layout.

Nothing in the readout is Qwen-specific. Any causal model with a chat template, a decoder at `model.model.layers`, a final `model.model.norm` and an `lm_head` should load. The heads need Yes, No, the digits and the letters to be single tokens. What has not been verified on other families is the cache fork and truncation; run `tests/test_cache.py` against a tiny config of that architecture first.

## Layout

```
src/dynajev/
  compile.py        typed questions and schemas -> field jobs, stages
  bind.py           field jobs + tokenizer -> field plans
  plan.py           IR: Branch, Read, Combine, Decode, FieldPlan, Plan
  executor.py       run a plan: stages, trie reads, combiners, decodes
  trie.py           trie prefill, cost model, prefix cache
  prompts.py        system text, blocks, answer boundary
  fit.py            affine and ridge corrections, layer choice
  heads.py          task signatures, stored heads
  backends/base.py  backend protocol
  backends/hf.py    transformers backend
  engine.py         decide / decide_batch / fit_heads
  server.py         FastAPI
  cli.py            serve, ask
web/                Next.js bench
eval/               labeled set and results
scripts/            eval.py, throughput.py, smoke.py
tests/
```
