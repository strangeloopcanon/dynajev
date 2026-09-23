# Dynajev

Build a Jev-style decision model out of any open-weight language model, per question, at request time.

[Jev](https://jevtypesafeai.com/docs), [Glance](https://glance.yohei.me/), [OpenJev](https://huggingface.co/openjev/openjev), [Laya](https://huggingface.co/convaiinnovations/laya) and [YOFO](https://arxiv.org/abs/2511.16600) each fix one answering architecture in advance, either hard-coded or trained. Dynajev compiles it from the question. Each typed question gets its own output head: which rows of the model's output matrix to read, where in the prompt, how many branches off a shared prefill, and how to combine the scores. The heads are attached to a stock model for one request and discarded. No weights change and nothing is trained.

The name is short for dynamic Jev. It is an independent project, not affiliated with TypeSafe.

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
  "refund": {"noul": 0.95},
  "queue":  {"choice": "billing and payments", "confidence": 0.63},
  "stars":  {"level": "2 stars", "score": 2.33},
  "flags":  {"flags": ["damage"], "probabilities": {"damage": 0.77, "refund request": 0.19, "delay": 0.23, "praise": 0.05}}
}
```

Abridged. The full response has every probability, the compiled head for each field, and token counts. Nothing above was generated.

## Question types

| `type` | Compiled head |
| --- | --- |
| `noul` | Yes rows minus No rows, sigmoid |
| `choice` | softmax over the option tokens, or over letters A, B, C… when options are phrases |
| `score` | softmax over digit rows; returns the level and the expected score |
| `flags` | one independent yes/no branch per flag |
| `quote` | short decode, checked verbatim against the text |
| `open` | ordinary chat completion (the only type that pays for generation) |

All closed questions about one text share a single prefill and branch at the answer. Pass a few labeled `examples` to fit a per-class bias or a small probe; it is kept only if held-out accuracy improves. `POST /api/decide_batch` runs one question set over many texts.

## Results

Qwen3.5-2B on a 4-core CPU, same model answering the same questions both ways. Details are in [`eval/RESULTS.md`](eval/RESULTS.md).

| | Dynajev | Model writes the answer |
| --- | --- | --- |
| Accuracy, 74 labeled questions | 71 | 72 |
| Median latency, one question | 259 ms | 369 ms |
| Six multi-field schemas | 5.8 s | 20.2 s |
| Throughput, 48 texts × 4 questions | 4.3 decisions/s | 0.9 decisions/s |
| Generated tokens, 74 questions | 0 | 222 |

The gain is speed and structure, not accuracy or total tokens. Both paths read the same first-token decision, and the prompt scaffolding costs about what the written answer would have. GPU numbers are not measured yet.

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

How requests compile, batching modes, server settings, and model support: [`docs/details.md`](docs/details.md).

MIT licensed.
