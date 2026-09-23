# Dynajev

Build a Jev-style decision model out of any open-weight language model, per question, at request time.

[Jev](https://jevtypesafeai.com/docs), [Glance](https://glance.yohei.me/), [OpenJev](https://huggingface.co/openjev/openjev), [Laya](https://huggingface.co/convaiinnovations/laya) and [YOFO](https://arxiv.org/abs/2511.16600) each settle on one answering architecture and push every question through it. Dynajev builds the architecture from the question. Each typed question compiles to its own output head: which rows of the model's output matrix it reads, where in the prompt it reads them, how many branches it runs off the shared prefill, and how the scores combine (a sigmoid, a softmax, an expected level, independent flags). The compiled heads are attached to a stock model for that one request, run in a single pass, and discarded. The model's weights never change and nothing is trained.

The name is short for dynamic Jev. It is an independent project and not affiliated with TypeSafe.

```bash
curl -s http://127.0.0.1:43124/api/decide -H 'content-type: application/json' -d '{
  "context": "The refund for order 4412 was issued Tuesday. The replacement mug arrived smashed. Two stars.",
  "questions": {
    "refund": {"type": "noul",   "instructions": "Was a refund issued?"},
    "queue":  {"type": "choice", "instructions": "Where should this go?", "options": ["billing and payments", "technical support", "account access"]},
    "stars":  {"type": "score",  "instructions": "How many stars?", "levels": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]},
    "flags":  {"type": "flags",  "instructions": "Which apply?", "options": ["damage", "refund request", "delay", "praise"]},
    "order":  {"type": "quote",  "instructions": "Quote the order number."},
    "next":   {"type": "open",   "instructions": "What should support do next?"}
  }
}'
```

```json
{
  "answers": {
    "refund": {"type": "noul",   "noul": 0.95},
    "queue":  {"type": "choice", "choice": "billing and payments", "probabilities": {"billing and payments": 0.63, "technical support": 0.33, "account access": 0.04}, "confidence": 0.63},
    "stars":  {"type": "score",  "level": "2 stars", "score": 2.33, "probabilities": {"1 star": 0.12, "2 stars": 0.63, "3 stars": 0.10, "4 stars": 0.11, "5 stars": 0.04}, "confidence": 0.63},
    "flags":  {"type": "flags",  "flags": ["damage"], "probabilities": {"damage": 0.77, "refund request": 0.19, "delay": 0.23, "praise": 0.05}},
    "order":  {"type": "quote",  "quote": "4412", "verbatim": true},
    "next":   {"type": "open",   "text": "Support should contact the customer to apologize for the damaged item and arrange a return or replacement."}
  },
  "prefill_tokens": 514,
  "generated_tokens": 24
}
```

Four of those six answers cost zero generated tokens. The quote decodes a few tokens and is checked against the text. The open question is an ordinary chat completion, because that is what an open question is.

## Why this is different

Every system in this space makes the same move: stop the model from writing and read a decision out of it directly. They differ in which decision structure they fix in advance.

| System | Structure it fixes | How it gets there |
| --- | --- | --- |
| Glance | Yes/No margin; one margin per option; digit expectation | Frozen vision-language model, fixed per question kind |
| Simple Jev, OpenJev shim | A letter or digit row at an unfinished `{"answer": "` | Frozen or lightly tuned causal model |
| OpenJev (Wortega), Laya | A trained head per task | Training |
| YOFO | Packed yes/no answer slots in one forward | Training |
| Jev | `noul`, `choice`, `score` | A model trained for it, hosted |

The observation behind Dynajev is that the right structure depends on the question, and that on a frozen causal model it can be chosen when the request arrives instead of being trained or fixed ahead of time. A yes/no question wants a two-row margin. Single-word options want a direct slice of those words; phrases want letters. A rating wants digit rows and an expectation rather than an argmax. Flags want independent yes/no branches, because a softmax would force exactly one. Several questions about the same text want one shared prefill and a branch per answer. With a few labels, a head can be swapped for a fitted one. None of this needs training, so it works on a model downloaded five minutes ago, and one request can mix all of it.

The mechanism underneath is small: a language model's output layer is already a classifier over its vocabulary, so a closed answer is a handful of rows of that matrix read at the right position. What Dynajev adds is the compiler that decides, per question, which rows, which position, how many branches, and which combination rule.

What you get compared with asking the model to write JSON:

- Every answer is one of your labels. There is no parsing and nothing to retry.
- Every answer has a probability, and `allowed_mass` tells you how much of the model's distribution was actually on your options. When that is low the answer is flagged.
- Latency is the prefill and nothing else. Measured below: about 1.4× faster on one question and 3.5× on a four-field schema, on CPU.
- Accuracy matches letting the model write the answer (71 vs 72 of 74 here), because it is the same first-token decision.
- Batches are rectangles. A readout has no decode loop, so many states × many questions is one padded forward per chunk with no ragged scheduling.

What you do not get: a smarter model. The readout reports what the trunk believes. A 2B model is a 2B model.

## The six question types

| `type` | Returns | How it is read |
| --- | --- | --- |
| `noul` | `noul`: probability the answer is yes | log-sum-exp of the Yes rows minus the No rows, then a sigmoid |
| `choice` | `choice`, `probabilities`, `confidence` | softmax over the option rows if each option is one token; otherwise options are lettered A, B, C and the letter rows are read |
| `score` | `level`, `score` (probability-weighted), `probabilities` | softmax over digit rows; digits follow the level numbering when levels are numbered ("1 star", "2 stars") |
| `flags` | `flags`, `probabilities` | one Yes/No margin per flag, each its own branch; not renormalized together |
| `quote` | `quote`, `verbatim` | short greedy decode stopping at the closing quote, then checked against the text |
| `open` | `text` | normal chat completion, up to 256 tokens |

The names follow Jev's so that anyone who knows that API can read this one. `noul` is a comparative yes/no (Yes versus No at the same position); Jev distinguishes that from an independent judgment, and so should you when you set thresholds.

If you send a single `question` with no type, the type is inferred from the shape: options mean choice, levels mean score, `exclusive: false` means flags, an opening auxiliary verb ("Is", "Does", "Has") means yes/no, anything else is open. The response says which rule fired. Declaring the type is the contract; inference is a convenience for quick calls.

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

`auto` picks `dense` on CUDA and `shared` on CPU, because that is what the measurements below say.

## What was measured

Everything below is on `Qwen/Qwen3.5-2B`, bfloat16, four CPU cores, reference kernels for the linear-attention layers. Numbers will be very different on a GPU with fused kernels; the ratios are the point.

### Read versus write

`eval/items.json` is 74 hand-labeled single questions (yes/no, one-token choice, phrase choice, rating, flags, quote), 6 schemas with 22 fields, and a 16-note tone set. `scripts/eval.py` runs every item as a read (Dynajev) and as a write (the same model generating the answer greedily, then parsed). Full tables in [`eval/RESULTS.md`](eval/RESULTS.md).

| | Read | Write |
| --- | --- | --- |
| 74 single questions, correct | 71 | 72 |
| median latency per question | 259 ms | 369 ms |
| tokens (prefill + generated) | 8063 + 0 | 6851 + 222 |
| 22 schema fields, correct | 17 | 17 as one JSON · 17 as one call per field |
| 6 schemas, total time | 5.8 s | 20.2 s one JSON · 10.7 s per field |
| 6 schemas, tokens | 1804 + 0 | 1103 + 165 one JSON · 2381 + 67 per field |

Accuracy is the same within noise because both paths read the same first-token decision: every closed type except flags scored identically, and on flags each path got different items wrong (read 5/8, write 6/8). The read is faster because there is no decode loop, and it is faster by more the more fields there are. Tokens are about even on single questions (the JSON scaffold in the read prompt costs about what two generated tokens cost) and, on schemas, sit between one JSON completion and one call per field.

### Batching

`scripts/throughput.py`, 48 states × 4 closed questions = 192 decisions:

| Method | Time | Decisions/s | Tokens |
| --- | --- | --- | --- |
| one `decide` per state (shared prefix within a state) | 45 s | 4.3 | 15,155 |
| `decide_batch`, `shared` mode | 46 s | 4.2 | 15,155 |
| `decide_batch`, `dense` mode, 8 rows per forward | 61 s | 3.1 | 33,221 |
| `decide_batch`, `dense` mode, 32 rows per forward | 79 s | 2.4 | 33,221 |
| the model writing one JSON per state | 209 s | 0.9 | 9,093 |

On this CPU the dense rectangle loses: it processes twice the tokens and the per-token throughput of these kernels gets worse, not better, as rows are added. Dense and shared agreed on 188 of 192 answers; the four that differ are bfloat16 rounding on near-even margins. That is a statement about four cores and reference kernels, not about the method. On a GPU a 150-token prefill of a 2B model leaves most of the chip idle, and throughput rises close to linearly with rows until compute saturates; that is the regime `dense` is for. It has not been measured here, so the README does not put a number on it.

The line that holds on any hardware is the last one: reading is about 5× the throughput of writing on the same model, with the same answers.

## Running it

Python 3.11+, Node 22 for the bench.

```bash
git clone https://github.com/strangeloopcanon/dynajev && cd dynajev
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # or the CUDA wheel
pip install -e ".[dev]"

dynajev serve --port 43124          # downloads Qwen/Qwen3.5-2B on first start (4.5 GB)
```

```bash
dynajev ask --context "The bicycle is red." --question "What color is the bicycle?" --options red blue green
```

Docker (CPU, weights baked into the image):

```bash
docker build -t dynajev . && docker run -p 43124:43124 dynajev
```

The web bench in `web/` shows the compiled heads, probabilities, and prompts for each request:

```bash
cd web && npm install && npm run dev      # http://127.0.0.1:43123, proxies /dynajev-api to the server
```

Tests do not download weights; they build tiny random models:

```bash
pytest
```

### Server

`dynajev serve` runs one model on one device, one forward at a time. `/api/ready` returns 503 until the weights are loaded; `/api/health` reports model, dtype, uptime, and cache statistics. Requests queue behind a bounded semaphore and get a 429 when it is full. Prefilled states are kept in a small LRU, so a second question about the same document only pays for its own suffix (a follow-up on a cached ticket answered in 165 ms here; a cold single question takes about 250 ms).

Environment variables: `DYNAJEV_MODEL`, `DYNAJEV_DTYPE` (`bfloat16` on CPU and `float16` on CUDA by default), `DYNAJEV_HOST`, `DYNAJEV_PORT`, `DYNAJEV_MAX_QUEUE`, `DYNAJEV_PREFIX_CACHE`, `DYNAJEV_CORS`.

## Models

The tested trunk is Qwen3.5-2B, loaded as its text-only decoder (the vision tower is not loaded). It is a hybrid: three Gated DeltaNet layers for every full-attention layer. Branching off a shared prefill has to carry recurrent and convolution states as well as key/value caches; the code deep-copies the whole cache, and a unit test checks that a batched branch equals a full forward on that layout. Larger Qwen3.5 checkpoints are the same layout and should work by setting `DYNAJEV_MODEL`.

The readout itself has nothing Qwen-specific in it. Any causal model with a chat template, a decoder at `model.model.layers`, and an `lm_head` will load; anything else is refused with a message pointing at the adapter spot. The heads need Yes, No, the digits, and the letters to be single tokens, which holds for every mainstream tokenizer. What has not been verified on other families is the cache branch. If you try one, run `tests/test_cache.py` against a tiny config of that architecture first.

## Limits

- Text only. A vision trunk would put image tokens in the shared prefix the same way Glance does; this build does not load one.
- Probabilities are renormalized over the allowed answers. Adding an option changes them. They are not calibrated frequencies unless you fit and check them on data the request did not see.
- The request-time fit is a small-n gate, not a training run. For a task you care about, collect labels and fit once.
- `quote` is a short generation with a verbatim check, not a hard copy mask. A mask was tried; it made the model quote from the first word of the text.
- Packed one-pass answer slots (YOFO, Laya) are not implemented. On a frozen model they read the template, not the answer.
- Dense batching is built for GPUs and has been measured only on a CPU, where it loses to the shared-prefix path.

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

MIT licensed.
