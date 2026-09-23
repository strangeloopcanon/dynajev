# Dynajev

Build a Jev-style decision model out of any open-weight language model, per question, at request time.

Each typed question compiles to its own output head on a stock model: which rows of the output matrix to read, where in the prompt, how many branches off a shared prefill, and how to combine the scores (a sigmoid, a softmax, an expected level, independent flags). The heads exist for one request and are discarded. No weights change and nothing is trained.

The name is short for dynamic [Jev](https://jevtypesafeai.com/docs). It is an independent project, not affiliated with TypeSafe.

## Example

```bash
curl -s http://127.0.0.1:43124/api/decide -H 'content-type: application/json' -d '{
  "context": "The refund for order 4412 was issued Tuesday. The replacement mug arrived smashed. Two stars.",
  "questions": {
    "refund": {"type": "noul",   "instructions": "Was a refund issued?"},
    "queue":  {"type": "choice", "instructions": "Where should this go?", "options": ["billing and payments", "technical support", "account access"]},
    "stars":  {"type": "score",  "instructions": "How many stars?", "levels": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]},
    "flags":  {"type": "flags",  "instructions": "Which apply?", "options": ["damage", "refund request", "delay", "praise"]}
  }
}'
```

```json
{
  "refund": {"noul": 0.96},
  "queue":  {"choice": "billing and payments", "confidence": 0.64},
  "stars":  {"level": "2 stars", "score": 2.05},
  "flags":  {"flags": ["damage"], "probabilities": {"damage": 0.87, "refund request": 0.15, "delay": 0.14, "praise": 0.00}}
}
```

Abridged. The full response has every probability, token counts, and a `plan`: the branches built for each question, which prompt segments were shared, how many layers ran, and how the scores were combined. Nothing above was generated.

## Question types

| `type` | Compiled head |
| --- | --- |
| `noul` | Yes rows minus No rows, sigmoid |
| `choice` | softmax over the option tokens, or over letters A, B, C… when options are phrases |
| `score` | softmax over digit rows; returns the level and the expected score |
| `flags` | one independent yes/no branch per flag |
| `quote` | short decode, checked verbatim against the text |
| `open` | ordinary chat completion |

The prompts of one request form a token trie; each shared segment is prefilled once when that is cheaper than recomputing it, and the rest run as one batch. `POST /api/decide_batch` runs one question set over many texts.

A question can depend on another: `depends_on: {"question": "damaged", "when": true}` skips it unless the answer matches, and `include_answers: ["damaged"]` continues from the parent's cached conversation so the model sees that answer. A `noul` with `criteria: {"true": "...", "false": "..."}` judges each description separately instead of reading one Yes-versus-No margin.

Pass a few labeled `examples` to fit a per-class bias or a ridge probe; it is kept only if leave-one-out accuracy holds. The fit also checks shallower layers and, if a probe there does as well, the question runs only that many layers. `save_heads: true` (or `POST /api/heads/fit`) keeps the head, keyed by the question, labels and model, and later requests with the same question use it.

## Results

Qwen3.5-2B on a 4-core CPU, same model answering the same questions both ways. Details are in [`eval/RESULTS.md`](eval/RESULTS.md).

| | Dynajev | Model writes the answer |
| --- | --- | --- |
| Accuracy, 74 labeled questions | 71 | 72 |
| Median latency, one question | 265 ms | 408 ms |
| Six multi-field schemas | 5.8 s | 22.7 s |
| Throughput, 48 texts × 4 questions | 4.0 decisions/s | 0.8 decisions/s |
| Generated tokens, 74 questions | 0 | 222 |

The gain is speed and structure, not accuracy or total tokens. Both paths read the same first-token decision, and the prompt scaffolding costs about what the written answer would have. A stored tone head that exits after layer 12 of 24 answers in 108 ms instead of 238 ms, but missed one of eight states the full-depth read got right. GPU numbers are not measured yet.

## Run it

```bash
git clone https://github.com/strangeloopcanon/dynajev && cd dynajev
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel
pip install -e ".[dev]"

dynajev serve --port 43124     # downloads Qwen/Qwen3.5-2B (4.5 GB) on first start
pytest                         # tiny random models, no download
```

A web bench that shows the compiled heads lives in `web/` (`npm install && npm run dev`, then open http://127.0.0.1:43123). Docker: `docker build -t dynajev . && docker run -p 43124:43124 dynajev`.

## Limits

- Text only. A vision trunk would put image tokens in the shared prefill, as Glance does; not built yet.
- Tested on Qwen3.5. Other causal models with a standard layout should load; check `tests/test_cache.py` against yours first.
- Probabilities are relative to the allowed answers, not calibrated frequencies.
- The model is not made smarter. The heads report what it already believes.

## Prior work

Dynajev is an extension of existing ideas, not a new method. The closest project is [Simple Jev](https://github.com/featherless-ai/simple-jev), which already serves `noul`, `choice` and `score` from a stock model by reading logits at `{"answer": "`, with one shared prefill and a batched branch per question. [typed-gguf](https://pypi.org/project/typed-gguf/) does similar for GGUF models, and [Glance](https://glance.yohei.me/) does it for vision-language models.

| Project | Model | Readout per question type | Shared prefill across questions |
| --- | --- | --- | --- |
| [Simple Jev](https://github.com/featherless-ai/simple-jev) | stock | fixed per type: letters, digits, 1–9 rating for `noul` | within a request |
| [typed-gguf](https://pypi.org/project/typed-gguf/) | stock GGUF | per type, with calibrated temperature | yes, with saved states |
| [Glance](https://glance.yohei.me/) | stock vision-language | yes/no, pick-one, digits | per image |
| [SGLang](https://docs.sglang.io/docs/references/frontend/choices_methods), [guidance](https://github.com/guidance-ai/guidance), [LMQL](https://lmql.ai/) | stock | option scoring, chosen by the programmer | SGLang |
| [decider](https://github.com/Mapika/decider), [Laya](https://huggingface.co/convaiinnovations/laya), [YOFO](https://arxiv.org/abs/2511.16600), Jev | trained | one learned format | packed slots or one pass |

What Dynajev adds on top: the head is chosen from how the labels tokenize; every question type shares one trie prefill; questions can depend on each other; labeled examples give a correction, and possibly an earlier exit layer, kept only if leave-one-out accuracy holds and stored for reuse; and there is a measured comparison against the same model writing its answers. Reading label-word logits goes back to [PET](https://arxiv.org/abs/2001.07676), and the bias fit is essentially [contextual calibration](https://arxiv.org/abs/2102.09690).

The plan IR, trie cost model, early exit, stored heads, backend interface, and server settings: [`docs/details.md`](docs/details.md).

MIT licensed.
