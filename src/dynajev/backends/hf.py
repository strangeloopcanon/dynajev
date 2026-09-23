"""Hugging Face transformers backend. The head never writes into these weights.

It implements `dynajev.backends.base.Backend`: prefill a token segment
into a cache, fork a cache, run many suffixes off one cache in a batch, each to
a chosen depth. Scoring uses only the unembedding rows the compiled head asked
for. A full-vocabulary matmul is used solely to report how much next-token mass
sat on those rows.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
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


class HFBackend:
    def __init__(self, model: Any, tokenizer: Any, model_id: str, device: str = "cpu"):
        self.model = model
        self.tok = tokenizer
        self.model_id = model_id
        self.device = device
        self.hidden_size = int(model.config.hidden_size)
        self.vocab_size = int(model.config.vocab_size)
        self.num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
        if not hasattr(model, "model") or not hasattr(model, "lm_head") or not hasattr(model.model, "layers"):
            raise ValueError(
                f"{model_id}: expected a decoder at model.model with .layers and an lm_head. "
                "Dynajev is tested on the Qwen3.5 layout; other layouts need a small adapter here."
            )
        self.model.eval()
        self.model.to(device)

    @classmethod
    def load(cls, model_id: str, device: str | None = None) -> "HFBackend":
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

    def continue_prompt(self, prompt_ids: list[int], answer: str, user_content: str, assistant_prefix: str) -> list[int] | None:
        """Finish the open assistant turn with `answer`, then ask a follow-up in a new user turn.

        The template is rendered with placeholder turns to learn the text that
        closes an assistant turn and opens the next one; that text is appended
        to `prompt_ids` as tokens, so the result starts with them exactly.
        Re-rendering the whole conversation would not: Qwen3.5 drops the
        empty think block from earlier assistant turns.
        """

        from dynajev.prompts import SYSTEM

        marks = ("\x00a\x00", "\x00b\x00", "\x00c\x00")
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": marks[0]},
            {"role": "assistant", "content": marks[1]},
            {"role": "user", "content": marks[2]},
        ]
        kwargs: dict[str, Any] = {"tokenize": False}
        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})
            kwargs["continue_final_message"] = True
        else:
            kwargs["add_generation_prompt"] = True
        try:
            text = self.tok.apply_chat_template(messages, **kwargs)
        except Exception:
            return None
        if not isinstance(text, str) or any(mark not in text for mark in marks):
            return None
        between = text[text.index(marks[1]) + len(marks[1]) : text.index(marks[2])]
        tail = text[text.index(marks[2]) + len(marks[2]) :]
        return list(prompt_ids) + self.token_ids(answer + between + user_content + tail)

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

    def fork(self, cache: Any) -> Any:
        """Copy a prefilled cache so a branch can extend it without touching the original.

        A deep copy keeps every layer's own class and state: keys/values for
        attention layers, conv and recurrent states for linear-attention layers
        (Qwen3.5 Gated DeltaNet). Rebuilding only K/V layers would silently
        drop the recurrent layers of a hybrid model.
        """

        return None if cache is None else copy.deepcopy(cache)

    def prefill(
        self, tokens: list[int], cache: Any = None, layers: tuple[int, ...] | None = None
    ) -> tuple[dict[int, torch.Tensor], Any]:
        """Extend `cache` (in place) by `tokens`; return the last position's hidden state per requested depth.

        A depth equal to the number of layers is the model's final, normed
        output. A smaller depth k runs only the first k decoder layers and
        returns the raw residual stream after layer k.
        """

        layers = self._layers(layers)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        taps, past = self._run(input_ids, cache, layers, [len(tokens) - 1], use_cache=True)
        return {k: taps[k][0] for k in layers}, past

    def prefill_rows(
        self, cache: Any, rows: list[list[int]], layers: list[tuple[int, ...]] | None = None, chunk_rows: int = 64
    ) -> list[dict[int, torch.Tensor]]:
        """Run many suffixes off one cache (or off nothing) as right-padded batches.

        Causality means the padding after a row's last real token cannot reach
        it, so no mask is needed and the hidden state is gathered at each row's
        own last position. `cache` is not modified.
        """

        if not rows:
            return []
        per_row = [self._layers(None if layers is None else layers[i]) for i in range(len(rows))]
        out: list[dict[int, torch.Tensor] | None] = [None] * len(rows)
        pad = self.tok.eos_token_id if getattr(self.tok, "eos_token_id", None) is not None else 0
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        chunk_rows = max(1, int(chunk_rows))
        for start in range(0, len(order), chunk_rows):
            index = order[start : start + chunk_rows]
            wanted = tuple(sorted({k for i in index for k in per_row[i]}))
            width = max(len(rows[i]) for i in index)
            batch = torch.tensor(
                [rows[i] + [pad] * (width - len(rows[i])) for i in index], dtype=torch.long, device=self.device
            )
            past = None if cache is None else self._expand(cache, len(index))
            taps, _ = self._run(batch, past, wanted, [len(rows[i]) - 1 for i in index], use_cache=past is not None)
            for row, i in enumerate(index):
                out[i] = {k: taps[k][row] for k in per_row[i]}
        return [item for item in out if item is not None]

    def read_dense(self, sequences: list[list[int]], chunk_rows: int = 32) -> list[torch.Tensor]:
        """Final hidden state for many unrelated prompts, one padded forward per chunk."""

        full = self.num_layers
        return [h[full] for h in self.prefill_rows(None, sequences, None, chunk_rows)]

    def _layers(self, layers: tuple[int, ...] | None) -> tuple[int, ...]:
        full = self.num_layers
        if not layers:
            return (full,)
        return tuple(sorted({min(max(1, int(k)), full) for k in layers}))

    def _run(
        self, input_ids: torch.Tensor, past: Any, layers: tuple[int, ...], positions: list[int], use_cache: bool
    ) -> tuple[dict[int, torch.Tensor], Any]:
        """One forward to the deepest requested layer; the hidden state of each row at its position, per layer.

        Shallower layers are captured with forward hooks on those decoder
        layers, gathering only the requested positions. `output_hidden_states`
        is not used: transformers installs its recording hooks once, on the
        layers present at the first call, which breaks under truncation.
        """

        depth = max(layers)
        base = self.model.model
        rows = torch.arange(input_ids.shape[0], device=self.device)
        index = torch.tensor(positions, dtype=torch.long, device=self.device)
        taps: dict[int, torch.Tensor] = {}
        handles = []
        for k in layers:
            if k == depth:
                continue

            def hook(_module: Any, _args: Any, output: Any, k: int = k) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                taps[k] = hidden[rows, index].detach().to(dtype=torch.float32).clone()

            handles.append(base.layers[k - 1].register_forward_hook(hook))
        try:
            # inference_mode: cache tensors with autograd history cannot be deep-copied.
            with torch.inference_mode(), self._truncated(depth):
                out = base(input_ids=input_ids, past_key_values=past, use_cache=use_cache, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        taps[depth] = out.last_hidden_state[rows, index].detach().to(dtype=torch.float32).clone()
        return taps, out.past_key_values

    @contextmanager
    def _truncated(self, depth: int):
        """Run only the first `depth` decoder layers, and skip the final norm.

        The model iterates `self.layers`, so a shorter ModuleList stops it early.
        Every layer keeps its own `layer_idx`, so it still writes its own cache
        slot; the slots of the skipped layers are left untouched.
        """

        base = self.model.model
        if depth >= self.num_layers:
            yield
            return
        layers, norm = base.layers, base.norm
        base.layers = torch.nn.ModuleList(list(layers)[:depth])
        base.norm = torch.nn.Identity()
        try:
            yield
        finally:
            base.layers = layers
            base.norm = norm

    def _expand(self, cache: Any, batch: int) -> Any:
        """Deep-copy a batch-1 cache and repeat every state tensor along the batch axis."""

        cloned = copy.deepcopy(cache)
        if batch == 1:
            return cloned
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
        return cloned

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
