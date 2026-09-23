# Read vs write on `Qwen/Qwen3.5-2B`

Same frozen model, same state, same question. Read = Readhead compiled head at the answer boundary. Write = ordinary greedy chat completion, parsed. CPU, bfloat16, 4 cores, reference DeltaNet kernels.

## Single questions

| Shape | n | Read acc | Write acc | Read tokens (prefill+gen) | Write tokens | Read ms (median) | Write ms (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boolean | 24 | 24/24 | 24/24 | 2194 (2194+0) | 2098 (2050+48) | 244 | 358 |
| choice_token | 12 | 12/12 | 12/12 | 1185 (1185+0) | 1121 (1096+25) | 238 | 350 |
| choice_phrase | 12 | 12/12 | 12/12 | 1344 (1344+0) | 1240 (1192+48) | 261 | 592 |
| ordinal | 12 | 12/12 | 12/12 | 1391 (1391+0) | 1251 (1219+32) | 269 | 474 |
| multilabel | 8 | 5/8 | 6/8 | 1343 (1343+0) | 804 (775+29) | 640 | 478 |
| extract | 6 | 6/6 | 6/6 | 606 (573+33) | 559 (519+40) | 576 | 581 |
| **all** | 74 | **71/74** | **72/74** | 8063 | 7073 | 259 | 369 |

Read total wall time 33.1 s vs write 35.9 s over 74 questions.

Where read and write disagree on correctness:

- `m01` read=['damage'] (wrong), write='damage, refund request' (ok)
- `m04` read=['refund request', 'delay'] (ok), write='refund request' (wrong)
- `m06` read=[] (wrong), write='delay' (ok)

Write outputs that did not parse to an allowed answer: 0 of 68.

## Schemas (several questions about one state)

| Variant | Fields correct | Tokens (prefill+gen) | Wall ms (sum) |
| --- | --- | --- | --- |
| Read: one shared prefill, a head per field | 17/22 | 1804 (1804+0) | 5820 |
| Write: one JSON completion | 17/22 | 1268 (1103+165) | 20232 |
| Write: one call per field | 17/22 | 2448 (2381+67) | 10680 |

- `s01` read missed: flags=['damage', 'refund request']
- `s02` read missed: locked_out=False
- `s03` read missed: need='a tracking update'
- `s03` json write missed: need='a tracking update'
- `s04` json write missed: flags=['refund', 'praise']
- `s05` read missed: priority='1'
- `s05` json write missed: priority=2
- `s06` read missed: flags=['damage', 'refund request', 'delay']
- `s06` json write missed: damaged=False, flags=['refund request', 'delay']

## Request-time fit (tone, 2 classes, 8 labeled examples per request, leave-one-out)

- Frozen slice: 16/16 correct, mean probability on the true label 0.989
- With fit: 16/16 correct, mean probability on the true label 0.989
- Fit chosen: {'zero_shot': 12, 'affine': 4}
- Median fit request time 2691 ms (8 extra example forwards)
