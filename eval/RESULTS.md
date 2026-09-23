# Read vs write on `Qwen/Qwen3.5-2B`

Same frozen model, same state, same question. Read = Dynajev compiled head at the answer boundary. Write = ordinary greedy chat completion, parsed. CPU, bfloat16, 4 cores, reference DeltaNet kernels.

## Single questions

| Shape | n | Read acc | Write acc | Read tokens (prefill+gen) | Write tokens | Read ms (median) | Write ms (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boolean | 24 | 24/24 | 24/24 | 2026 (2026+0) | 2098 (2050+48) | 259 | 384 |
| choice_token | 12 | 12/12 | 12/12 | 998 (998+0) | 1121 (1096+25) | 244 | 384 |
| choice_phrase | 12 | 12/12 | 12/12 | 1212 (1212+0) | 1240 (1192+48) | 265 | 647 |
| ordinal | 12 | 12/12 | 12/12 | 1271 (1271+0) | 1251 (1219+32) | 267 | 516 |
| multilabel | 8 | 5/8 | 6/8 | 1023 (1023+0) | 804 (775+29) | 583 | 523 |
| extract | 6 | 6/6 | 6/6 | 541 (489+52) | 559 (519+40) | 886 | 630 |
| **all** | 74 | **71/74** | **72/74** | 7071 | 7073 | 265 | 408 |

Read total wall time 27.7 s vs write 38.1 s over 74 questions.

Where read and write disagree on correctness:

- `m02` read=['praise'] (wrong), write='delay, praise' (ok)
- `m04` read=['refund request', 'delay'] (ok), write='refund request' (wrong)
- `m06` read=[] (wrong), write='delay' (ok)

Write outputs that did not parse to an allowed answer: 0 of 68.

## Schemas (several questions about one state)

| Variant | Fields correct | Tokens (prefill+gen) | Wall ms (sum) |
| --- | --- | --- | --- |
| Read: one shared prefill, a head per field | 19/22 | 1720 (1720+0) | 5822 |
| Write: one JSON completion | 17/22 | 1268 (1103+165) | 22746 |
| Write: one call per field | 17/22 | 2448 (2381+67) | 11988 |

- `s03` read missed: need='a tracking update'
- `s03` json write missed: need='a tracking update'
- `s04` json write missed: flags=['refund', 'praise']
- `s05` read missed: priority='1'
- `s05` json write missed: priority=2
- `s06` read missed: flags=['refund request', 'delay']
- `s06` json write missed: damaged=False, flags=['refund request', 'delay']

## Request-time fit (tone, 2 classes, 8 labeled examples per request, leave-one-out)

- Frozen slice: 16/16 correct, mean probability on the true label 0.974
- With fit: 16/16 correct, mean probability on the true label 1.000
- Fit chosen: {'ridge_probe': 16}
- Median fit request time 1794 ms (8 extra example forwards)

## Stored head with early exit (tone, fitted once on 8 states, applied to the other 8)

- Fit: ridge_probe, exits after layer 12 of 24; one fit call took 1720 ms
- Layer scan (leave-one-out on the 8 fit states): layer 4: acc 0.12, nll 0.798, layer 8: acc 0.88, nll 0.440, layer 12: acc 1.00, nll 0.000; full-depth head acc 1.00, nll 0.005
- Zero-shot read, full depth: 8/8 correct, median 238 ms
- Stored head: 7/8 correct, median 108 ms (2.20x)
- Times are the best of 3 runs per state, one state per request.
- The shallow probe is faster but less accurate than the frozen full-depth read on these states. Its leave-one-out check ran on only 8 fit states and did not catch that.
