# Dynajev

Turn any open-weight LLM into a Jev-style decision model, per question, at request time.

Each typed question compiles its own output head: which rows of the output matrix to read, where in the prompt, how many branches off a shared prefill, how many layers to run, and how to combine the scores. The weights stay frozen. Closed questions generate nothing.

Dynajev is short for dynamic [Jev](https://jevtypesafeai.com/docs).

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

The full response also returns the `plan`: the architecture built for this request.

## Question types

| `type` | Head |
| --- | --- |
| `noul` | Yes rows minus No rows, sigmoid; or `criteria` judged independently |
| `choice` | softmax over option tokens, or letters when options are phrases |
| `score` | digit rows, expected level |
| `flags` | independent yes/no per flag |
| `quote` | short decode, verified against the text |
| `open` | plain generation |

- **Shared prefill.** A request's prompts form a token trie; shared segments run once.
- **Dependencies.** `depends_on` skips a question unless an earlier answer matches; `include_answers` continues from the parent's cache.
- **Learned heads.** Labeled `examples` fit a per-task bias or probe and pick the shallowest layer that holds accuracy. Saved heads apply to later requests automatically.
- **Batch.** `POST /api/decide_batch` runs one question set over many texts.

## Results

Qwen3.5-2B, 4-core CPU, same model answering both ways ([details](eval/RESULTS.md)).

| | Dynajev | Model writes the answer |
| --- | --- | --- |
| Accuracy, 74 questions | 71 | 72 |
| Latency, one question | 265 ms | 408 ms |
| Six multi-field schemas | 5.8 s | 22.7 s |
| Throughput, 48 texts × 4 questions | 4.0 decisions/s | 0.8 decisions/s |
| Generated tokens | 0 | 222 |

Early exit on a learned tone head: 108 ms vs 238 ms at full depth (7/8 vs 8/8 correct).

## Run it

```bash
git clone https://github.com/strangeloopcanon/dynajev && cd dynajev
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel
pip install -e ".[dev]"

dynajev serve --port 43124     # pulls Qwen/Qwen3.5-2B on first start
pytest
```

Web bench: `cd web && npm install && npm run dev` → http://127.0.0.1:43123. Docker: `docker build -t dynajev . && docker run -p 43124:43124 dynajev`.

Text only for now; tested on Qwen3.5.

## Related

Builds on [Simple Jev](https://github.com/featherless-ai/simple-jev), [typed-gguf](https://pypi.org/project/typed-gguf/), [Glance](https://glance.yohei.me/), [decider](https://github.com/Mapika/decider), [Laya](https://huggingface.co/convaiinnovations/laya), [YOFO](https://arxiv.org/abs/2511.16600), [PET](https://arxiv.org/abs/2001.07676) and [contextual calibration](https://arxiv.org/abs/2102.09690). Internals: [`docs/details.md`](docs/details.md).

MIT licensed.
