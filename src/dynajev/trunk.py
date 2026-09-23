"""Frozen causal trunk. The head never writes into these weights.

`read` prefills the shared token prefix once and branches each answer boundary
off a copy of the cache. Scoring uses only the unembedding rows the compiled
head asked for. A full-vocabulary matmul is used solely to report how much
next-token mass sat on those rows.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import torch


def trim_repetition(text: str) -> str:
    """Cut a decoded span at the first repeated word cycle.

    A masked greedy decode can orbit a handful of context tokens. The span up
    to the first repeat is the part that was actually copied.
    """

    words = text.split()
    count = len(words)
    for size in range(2, count // 2 + 1):
        for start in range(0, count - 2 * size + 1):
            if words[start : start + size] == words[start + size : start + 2 * size]:
                return " ".join(words[: start + size]).rstrip(",").strip()
    return text.strip().rstrip(",").strip()


def _pick_dtype(device: str) -> torch.dtype:
    """DYNAJEV_DTYPE overrides. Default: float16 on CUDA, bfloat16 on CPU.

    bfloat16 halves the memory of a 2B trunk on CPU (about 4.5 GB instead of 9)
    and every readout is computed in float32 from the last hidden state anyway.
    """

    import os

    named = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    wanted = os.environ.get("DYNAJEV_DTYPE", "").lower()
    if wanted in named:
        return named[wanted]
    return torch.float16 if device == "cuda" else torch.bfloat16


def common_prefix_len(sequences: list[list[int]]) -> int:
    if not sequences:
        return 0
    length = 0
    for tokens in zip(*sequences):
        first = tokens[0]
        if any(token != first for token in tokens):
            break
        length += 1
    return length


class Trunk:
    def __init__(self, model: Any, tokenizer: Any, model_id: str, device: str = "cpu"):
        self.model = model
        self.tok = tokenizer
        self.model_id = model_id
        self.device = device
        self.hidden_size = int(model.config.hidden_size)
        self.vocab_size = int(model.config.vocab_size)
        if not hasattr(model, "model") or not hasattr(model, "lm_head") or not hasattr(model.model, "layers"):
            raise ValueError(
                f"{model_id}: expected a decoder at model.model with .layers and an lm_head. "
                "Dynajev is tested on the Qwen3.5 layout; other layouts need a small adapter here."
            )
        self.model.eval()
        self.model.to(device)
        # Prefilled states kept across requests, keyed by their exact token prefix.
        self.prefix_cache_size = 0
        self._prefix_cache: OrderedDict[tuple[int, ...], tuple[torch.Tensor, Any]] = OrderedDict()
        self._prefix_hits = 0
        self._prefix_misses = 0
        self.last_read_cached = False

    def prefix_cache_stats(self) -> dict[str, int]:
        return {
            "entries": len(self._prefix_cache),
            "capacity": self.prefix_cache_size,
            "hits": self._prefix_hits,
            "misses": self._prefix_misses,
        }

    @classmethod
    def load(cls, model_id: str, device: str | None = None) -> "Trunk":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=_pick_dtype(device), low_cpu_mem_usage=True
        )
        return cls(model, tokenizer, model_id, device)

    def token_ids(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def single_token(self, text: str) -> int | None:
        ids = self.token_ids(text)
        if len(ids) == 1:
            return ids[0]
        return None

    def sanitize(self, context: str) -> str:
        specials = list(getattr(self.tok, "all_special_tokens", []) or [])
        for token in specials:
            context = context.replace(token, "")
        return context

    def encode_prompt(self, user_content: str, assistant_prefix: str, system: str | None = None) -> list[int]:
        from dynajev.prompts import SYSTEM

        messages = [
            {"role": "system", "content": SYSTEM if system is None else system},
            {"role": "user", "content": user_content},
        ]
        if not assistant_prefix:
            # An empty assistant turn is a plain generation prompt, not a boundary to continue.
            return list(self._apply(messages, continue_final=False, generation_prompt=True) or [])
        messages.append({"role": "assistant", "content": assistant_prefix})
        ids = self._apply(messages, continue_final=True)
        if ids is None:
            ids = self._apply(messages[:2], continue_final=False, generation_prompt=True)
            ids = list(ids) + self.token_ids(assistant_prefix)
        return list(ids)

    def _apply(self, messages: list[dict[str, str]], continue_final: bool, generation_prompt: bool = False) -> list[int] | None:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": generation_prompt,
        }
        if continue_final:
            kwargs["continue_final_message"] = True
        try:
            ids = self.tok.apply_chat_template(messages, **kwargs)
        except Exception:
            return None
        if isinstance(ids, dict) or hasattr(ids, "keys") and "input_ids" in ids.keys():
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token) for token in ids]

    def read(self, sequences: list[list[int]], anchor: int | None = None) -> tuple[list[torch.Tensor], int]:
        """Hidden state at the last position of each sequence.

        The common prefix is prefilled once and every sequence branches off it in
        one batched forward. With `anchor` (the token length of the state block,
        which is the same across requests about the same state) and a non-zero
        `prefix_cache_size`, the prefill at that anchor is kept and reused by
        later requests, so a second question about the same document costs only
        its own suffix.
        """

        self.last_read_cached = False
        if not sequences:
            return [], 0
        shared = common_prefix_len(sequences)
        with torch.inference_mode():
            use_anchor = anchor is not None and 0 < anchor <= shared and self.prefix_cache_size > 0
            if use_anchor:
                shared = int(anchor)
            if shared == 0:
                return [self._hidden(seq, None) for seq in sequences], 0
            key = tuple(sequences[0][:shared])
            cached = self._prefix_cache.get(key) if use_anchor else None
            if cached is not None:
                shared_hidden, past = cached
                self._prefix_cache.move_to_end(key)
                self._prefix_hits += 1
                self.last_read_cached = True
            else:
                shared_hidden, past = self._forward(sequences[0][:shared], None)
                if use_anchor:
                    self._prefix_misses += 1
                    self._prefix_cache[key] = (shared_hidden, past)
                    while len(self._prefix_cache) > self.prefix_cache_size:
                        self._prefix_cache.popitem(last=False)
            hiddens: list[torch.Tensor | None] = [None] * len(sequences)
            branch_index = []
            suffixes = []
            for index, seq in enumerate(sequences):
                if len(seq) == shared:
                    hiddens[index] = shared_hidden.clone()
                else:
                    branch_index.append(index)
                    suffixes.append(seq[shared:])
            if suffixes:
                batched = self._branch_batch(past, suffixes)
                if batched is None:
                    for index, suffix in zip(branch_index, suffixes):
                        cloned = self._clone_past(past)
                        hiddens[index] = self._hidden(suffix, cloned) if cloned is not None else self._hidden(sequences[index], None)
                else:
                    for index, hidden in zip(branch_index, batched):
                        hiddens[index] = hidden
            return [h for h in hiddens if h is not None], shared

    def read_dense(self, sequences: list[list[int]], chunk_rows: int = 32) -> list[torch.Tensor]:
        """Last-position hidden state for many unrelated prompts, one padded forward per chunk.

        Right padding, no cache, no mask: a causal model cannot see the padding
        that follows a row's last real token, so gathering at that index is exact.
        """

        if not sequences:
            return []
        pad = self.tok.eos_token_id if getattr(self.tok, "eos_token_id", None) is not None else 0
        out: list[torch.Tensor | None] = [None] * len(sequences)
        chunk_rows = max(1, int(chunk_rows))
        # Chunk rows of similar length together so padding does not grow with the chunk.
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        with torch.inference_mode():
            for start in range(0, len(order), chunk_rows):
                index = order[start : start + chunk_rows]
                chunk = [sequences[i] for i in index]
                width = max(len(s) for s in chunk)
                rows = [s + [pad] * (width - len(s)) for s in chunk]
                input_ids = torch.tensor(rows, dtype=torch.long, device=self.device)
                hidden = self.model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
                for row, (i, s) in enumerate(zip(index, chunk)):
                    out[i] = hidden[row, len(s) - 1, :].detach().to(dtype=torch.float32).clone()
        return [h for h in out if h is not None]

    def _branch_batch(self, past: Any, suffixes: list[list[int]]) -> list[torch.Tensor] | None:
        """Run every branch suffix in one forward over a batch-expanded copy of the shared cache.

        Suffixes are right-padded. Causality means the padding after a branch's
        last real token cannot reach it, so no mask is needed; the hidden state
        is gathered at each branch's own last position. Returns None when the
        cache layout is unknown, and the caller falls back to one forward per branch.
        """

        batch = len(suffixes)
        expanded = self._expand_past(past, batch)
        if expanded is None:
            return None
        width = max(len(s) for s in suffixes)
        pad = self.tok.eos_token_id if getattr(self.tok, "eos_token_id", None) is not None else 0
        rows = [s + [pad] * (width - len(s)) for s in suffixes]
        input_ids = torch.tensor(rows, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            out = self.model.model(input_ids=input_ids, past_key_values=expanded, use_cache=True, return_dict=True)
        last = out.last_hidden_state
        return [last[i, len(s) - 1, :].detach().to(dtype=torch.float32).clone() for i, s in enumerate(suffixes)]

    def _expand_past(self, past: Any, batch: int) -> Any:
        """Deep-copy a batch-1 cache and repeat every state tensor along the batch axis."""

        cloned = self._clone_past(past)
        if cloned is None or not hasattr(cloned, "layers"):
            return None
        try:
            for layer in cloned.layers:
                for name in ("keys", "values"):
                    tensor = getattr(layer, name, None)
                    if torch.is_tensor(tensor) and tensor.dim() >= 1 and tensor.shape[0] == 1:
                        setattr(layer, name, tensor.repeat_interleave(batch, dim=0))
                for name in ("conv_states", "recurrent_states"):
                    states = getattr(layer, name, None)
                    if isinstance(states, dict):
                        for key, tensor in states.items():
                            if torch.is_tensor(tensor) and tensor.dim() >= 1 and tensor.shape[0] == 1:
                                states[key] = tensor.repeat_interleave(batch, dim=0)
                    elif torch.is_tensor(states) and states.dim() >= 1 and states.shape[0] == 1:
                        setattr(layer, name, states.repeat_interleave(batch, dim=0))
        except Exception:
            return None
        return cloned

    def _hidden(self, ids: list[int], past: Any) -> torch.Tensor:
        hidden, _ = self._forward(ids, past)
        return hidden

    def _forward(self, ids: list[int], past: Any) -> tuple[torch.Tensor, Any]:
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": True, "return_dict": True}
        if past is not None:
            kwargs["past_key_values"] = past
        base = self.model.model
        # inference_mode here, not only in read(): cache tensors with autograd
        # history cannot be deep-copied, and a branch must be able to clone them.
        with torch.inference_mode():
            out = base(**kwargs)
        hidden = out.last_hidden_state[0, -1, :].detach().to(dtype=torch.float32).clone()
        return hidden, out.past_key_values

    def _clone_past(self, past: Any) -> Any:
        """Copy a prefilled cache so a branch can extend it without touching the original.

        A deep copy keeps every layer's own class and state: keys/values for
        attention layers, conv and recurrent states for linear-attention layers
        (Qwen3.5 Gated DeltaNet). Rebuilding only K/V layers would silently
        drop the recurrent layers of a hybrid model, so that path is gone. If
        the copy fails, the caller falls back to a full forward.
        """

        if past is None:
            return None
        try:
            if hasattr(past, "layers"):
                cloned = copy.deepcopy(past)
                if len(getattr(cloned, "layers", [])) != len(past.layers):
                    return None
                return cloned
            if hasattr(past, "to_legacy_cache"):
                legacy = past.to_legacy_cache()
                cloned_legacy = tuple(tuple(tensor.detach().clone() for tensor in layer) for layer in legacy)
                cache_cls = type(past)
                if hasattr(cache_cls, "from_legacy_cache"):
                    return cache_cls.from_legacy_cache(cloned_legacy)
            if isinstance(past, tuple):
                return tuple(tuple(tensor.detach().clone() for tensor in layer) for layer in past)
        except Exception:
            return None
        return None

    def class_logits(self, hidden: torch.Tensor, groups: list[list[int]]) -> tuple[list[float], float]:
        weight = self.model.lm_head.weight
        bias = getattr(self.model.lm_head, "bias", None)
        hidden_row = hidden.detach().to(dtype=torch.float32).view(-1)
        scores: list[float] = []
        allowed: list[int] = []
        for group in groups:
            rows = weight[group].detach().to(dtype=torch.float32)
            token_scores = rows @ hidden_row
            if bias is not None:
                token_scores = token_scores + bias[group].detach().to(dtype=torch.float32)
            scores.append(torch.logsumexp(token_scores, dim=0).item())
            allowed.extend(group)
        # The allowed-mass report needs the whole distribution once. Run that
        # matvec in the head's own dtype: converting a 248k x 2048 weight to
        # float32 per call costs more than the trunk forward itself.
        full = torch.nn.functional.linear(hidden_row.to(dtype=weight.dtype), weight.detach(), None if bias is None else bias.detach()).to(dtype=torch.float32)
        unique = list(dict.fromkeys(allowed))
        mass = torch.exp(torch.logsumexp(full[unique], dim=0) - torch.logsumexp(full, dim=0)).item()
        return scores, float(mass)

    def prototype_logits(self, hidden: torch.Tensor, pieces: list[list[int]]) -> list[float]:
        weight = self.model.lm_head.weight.detach()
        hidden_row = hidden.detach().to(dtype=torch.float32).view(-1)
        scores: list[float] = []
        for ids in pieces:
            prototype = weight[ids].to(dtype=torch.float32).mean(dim=0)
            scores.append(torch.dot(hidden_row, prototype).item())
        return scores

    def as_vector(self, hidden: torch.Tensor) -> list[float]:
        return hidden.detach().to(dtype=torch.float32).view(-1).tolist()

    def generate(
        self, prompt_ids: list[int], context: str, extractive: bool, mode: str = "short"
    ) -> tuple[str, int]:
        """Short greedy decode at the opened JSON string.

        `extractive` only changes the stop set: a quote may legitimately contain a
        period ("M. Ortiz", "$1,240.00"), so it stops at the closing quote, a
        newline, or end of turn. Whether the text is verbatim is checked by the
        caller, not forced by a mask: a hard copy mask made the model start
        quoting from the first word of the state.
        """

        if mode == "open":
            return self._generate_open(prompt_ids)
        eos = [self.tok.eos_token_id]
        # The closing quote often merges with what follows it ('"}', '",'), so
        # those merged tokens are stops too, and the decoded text is trimmed after.
        closers = ('"', '"}', '",', '"\n', "\n")
        stops = closers if extractive else closers + (".", "。")
        for stop in stops:
            tid = self.single_token(stop)
            if tid is not None and tid not in eos:
                eos.append(tid)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model.generate(
                prompt,
                max_new_tokens=24,
                do_sample=False,
                # A quote must be free to repeat n-grams that occur in the prompt;
                # that is what quoting is. Loops are cut afterwards by trim_repetition.
                no_repeat_ngram_size=0 if extractive else 3,
                eos_token_id=eos,
                pad_token_id=self.tok.eos_token_id,
            )
        new_ids = output[0, prompt.shape[1] :].tolist()
        text = self.tok.decode(new_ids, skip_special_tokens=True).strip()
        for closer in ('"}', '",', '"'):
            if text.endswith(closer):
                text = text[: -len(closer)]
        text = text.strip().strip('"').strip(" .,;:")
        text = trim_repetition(text)
        return text, len(new_ids)

    def _generate_open(self, prompt_ids: list[int], max_new_tokens: int = 256) -> tuple[str, int]:
        """A normal chat completion: greedy, stops only at the model's end-of-turn."""
        eos = [self.tok.eos_token_id]
        for special in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
            tid = self.single_token(special)
            if tid is not None and tid not in eos:
                eos.append(tid)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=eos,
                pad_token_id=self.tok.eos_token_id,
            )
        new_ids = output[0, prompt.shape[1] :].tolist()
        text = self.tok.decode(new_ids, skip_special_tokens=True).strip()
        return text, len(new_ids)
