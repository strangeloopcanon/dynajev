"""What the compiler and the trie reader need from a model runtime.

A backend owns the frozen model and its tokenizer. Caches are opaque to the
rest of the code: the reader only prefills into them, forks them, and runs
batches of suffixes off them. A depth `k` in `layers` means "run the first k
decoder layers and return the residual stream there"; `num_layers` means the
model's final, normed output.
"""

from __future__ import annotations

from typing import Any, Protocol


class Backend(Protocol):
    model_id: str
    hidden_size: int
    vocab_size: int
    num_layers: int
    device: str

    def sanitize(self, text: str) -> str:
        """Strip anything that would be read as a control token."""

    def token_ids(self, text: str) -> list[int]: ...

    def single_token(self, text: str) -> int | None:
        """The id when `text` is exactly one token, else None."""

    def encode_prompt(self, user_content: str, assistant_prefix: str, system: str | None = None) -> list[int]:
        """Chat-template tokens ending at an unfinished assistant turn (or a generation prompt)."""

    def continue_prompt(self, prompt_ids: list[int], answer: str, user_content: str, assistant_prefix: str) -> list[int] | None:
        """`prompt_ids` with its open assistant turn finished by `answer`, then a new user turn.

        Must start with `prompt_ids` exactly, so the new prompt forks that
        prompt's cache. None when the template cannot be continued.
        """

    def prefill(self, tokens: list[int], cache: Any = None, layers: tuple[int, ...] | None = None) -> tuple[dict[int, Any], Any]:
        """Extend `cache` by `tokens` (in place); hidden state at the last position per requested depth."""

    def fork(self, cache: Any) -> Any:
        """An independent copy of `cache`."""

    def prefill_rows(
        self, cache: Any, rows: list[list[int]], layers: list[tuple[int, ...]] | None = None, chunk_rows: int = 64
    ) -> list[dict[int, Any]]:
        """Many suffixes off one cache (not modified), batched; last-position hidden per row and depth."""

    def read_dense(self, sequences: list[list[int]], chunk_rows: int = 32) -> list[Any]:
        """Final hidden state of many unrelated prompts."""

    def class_logits(self, hidden: Any, groups: list[list[int]]) -> tuple[list[float], float]:
        """Log-sum-exp of the unembedding rows of each group, and the next-token mass on all of them."""

    def prototype_logits(self, hidden: Any, pieces: list[list[int]]) -> list[float]: ...

    def as_vector(self, hidden: Any) -> list[float]: ...

    def generate(self, prompt_ids: list[int], context: str, extractive: bool, mode: str = "short") -> tuple[str, int]:
        """Greedy decode from `prompt_ids`; returns text and the number of generated tokens."""
