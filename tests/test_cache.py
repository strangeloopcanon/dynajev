import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from dynajev.plan import Branch
from dynajev.trie import TrieReader, build_trie
from dynajev.backends.hf import HFBackend

IDS = [7, 11, 3, 9, 4, 8, 2, 6, 5, 10]
FULL = 4


def _tiny_qwen35() -> Qwen3_5ForCausalLM:
    # Qwen3.5 layout: linear-attention (Gated DeltaNet) layers with conv and
    # recurrent state, plus full-attention layers with K/V. The clone must carry both.
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=FULL,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=128,
        attn_implementation="eager",
    )
    return Qwen3_5ForCausalLM(config).eval()


def _trunk() -> HFBackend:
    return HFBackend(_tiny_qwen35(), _Stub(), model_id="tiny-qwen3.5")


def _full(trunk: HFBackend, seq: list[int]) -> torch.Tensor:
    return trunk.prefill(seq)[0][FULL]


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a, b, atol=1e-4, rtol=1e-4)


def test_forked_cache_matches_a_full_forward_on_hybrid_deltanet():
    trunk = _trunk()
    full = _full(trunk, IDS)
    _, past = trunk.prefill(IDS[:6])
    branched = trunk.prefill(IDS[6:], trunk.fork(past))[0][FULL]
    assert _close(full, branched)
    # Rows of different lengths off one cache, one batched forward.
    others = [[1, 2, 3], [12, 13, 14, 15, 16, 17, 18], IDS[6:]]
    rows = trunk.prefill_rows(past, others)
    for taps, suffix in zip(rows, others):
        assert _close(taps[FULL], _full(trunk, IDS[:6] + suffix))
    # The original cache is untouched by the branches, so another fork is clean.
    again = trunk.prefill(IDS[6:], trunk.fork(past))[0][FULL]
    assert _close(full, again)


def _branches(sequences: list[list[int]], depths: list[int | None] | None = None) -> list[Branch]:
    depths = depths or [None] * len(sequences)
    return [Branch(id=f"b{i}", field="f", block="", tokens=seq, depth=d) for i, (seq, d) in enumerate(zip(sequences, depths))]


# A state prefix, then a question stem shared by four flag branches, then the
# labels; plus two other questions, one of which is a prefix of a longer branch.
STATE = [7, 11, 3, 9, 4, 8, 2, 6]
STEM = [20, 21, 22, 23, 24, 25]
TAIL = [30, 31, 32]
TRIE_SEQUENCES = [
    STATE + STEM + [40] + TAIL,
    STATE + STEM + [41, 42] + TAIL,
    STATE + STEM + [43] + TAIL,
    STATE + STEM + [44, 45, 46] + TAIL,
    STATE + [50, 51, 52] + TAIL,
    STATE + [50, 51],
    STATE + [60] + TAIL,
]


def test_trie_reads_match_full_forwards_when_every_shared_segment_is_split():
    trunk = _trunk()
    reader = TrieReader(trunk, overhead_tokens=0)
    hiddens, record = reader.read(_branches(TRIE_SEQUENCES))
    for i, seq in enumerate(TRIE_SEQUENCES):
        assert _close(hiddens[f"b{i}"], _full(trunk, seq)), f"branch {i} differs"
    # State once, stem once, the [50, 51] node once, then the leaves.
    assert record["segments"][0]["tokens"] == len(STATE)
    assert any(seg["tokens"] == len(STEM) and seg["rows"] == 4 for seg in record["segments"])
    naive = sum(len(s) for s in TRIE_SEQUENCES)
    assert record["naive_tokens"] == naive
    trie_tokens = len(STATE) + len(STEM) + (1 + 2 + 1 + 3) + 4 * len(TAIL) + 2 + 1 + len(TAIL) + 1 + len(TAIL)
    assert record["processed_tokens"] == trie_tokens < naive


def test_trie_reads_match_full_forwards_with_the_default_cost_model():
    trunk = _trunk()
    reader = TrieReader(trunk)
    hiddens, record = reader.read(_branches(TRIE_SEQUENCES))
    for i, seq in enumerate(TRIE_SEQUENCES):
        assert _close(hiddens[f"b{i}"], _full(trunk, seq))
    # Short segments are not worth a forward of their own, so they ride in the batch.
    assert record["forwards"] <= 2


def test_trie_groups_sequences_by_their_divergence_points():
    root = build_trie(TRIE_SEQUENCES, [FULL] * len(TRIE_SEQUENCES))
    assert root.tokens == STATE
    assert root.rows == 7
    stem = next(c for c in root.children if c.tokens[:1] == [20])
    assert stem.tokens == STEM and stem.rows == 4
    fifty = next(c for c in root.children if c.tokens[:1] == [50])
    assert fifty.tokens == [50, 51] and fifty.ends == [5]


def test_kept_branches_let_a_later_stage_fork_their_exact_cache():
    trunk = _trunk()
    reader = TrieReader(trunk, overhead_tokens=0)
    parent = STATE + STEM + TAIL
    first = _branches([parent, STATE + [60] + TAIL])
    first[0].keep = True
    kept: dict = {}
    reader.read(first, None, kept)
    assert tuple(parent) in kept
    child = parent + [70, 71, 72, 73]
    hiddens, record = reader.read(_branches([child, STATE + [61]]), None, kept)
    assert _close(hiddens["b0"], _full(trunk, child))
    assert _close(hiddens["b1"], _full(trunk, STATE + [61]))
    assert record["cached_tokens"] == len(parent)


def test_prefix_store_reuses_the_state_prefill_across_requests():
    trunk = _trunk()
    reader = TrieReader(trunk, prefix_cache=2)
    anchor = STATE + [99]
    first, record = reader.read(_branches([STATE + [1, 2], STATE + [3, 4, 5]]), anchor)
    assert reader.store.stats()["misses"] == 1 and record["cached_tokens"] == 0
    second, record = reader.read(_branches([STATE + [6], STATE + [1, 2]]), anchor)
    assert reader.store.stats()["hits"] == 1 and record["cached_tokens"] == len(STATE)
    assert _close(second["b0"], _full(trunk, STATE + [6]))
    assert _close(second["b1"], first["b0"])
    # A deeper request than the cached one cannot reuse a shallow prefill.
    shallow, _ = reader.read(_branches([[5, 5, 5, 1], [5, 5, 5, 2]], [2, 2]), [5, 5, 5, 0])
    reader.read(_branches([[5, 5, 5, 1], [5, 5, 5, 2]]), [5, 5, 5, 0])
    assert reader.store.stats()["misses"] == 3
    assert reader.store.stats()["entries"] == 2


def test_dense_read_matches_full_forwards_row_by_row():
    trunk = _trunk()
    rows = [IDS, IDS[:4], [3, 1, 4, 1, 5, 9, 2, 6], [7]]
    dense = trunk.read_dense(rows, chunk_rows=3)
    assert len(dense) == 4
    for hidden, seq in zip(dense, rows):
        assert _close(hidden, _full(trunk, seq))


def test_unknown_layout_is_refused():
    class Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"hidden_size": 4, "vocab_size": 8})()

    try:
        HFBackend(Bare(), _Stub(), model_id="bare")
    except ValueError as exc:
        assert "model.model" in str(exc)
    else:
        raise AssertionError("a model without the expected layout was accepted")


class _Stub:
    eos_token_id = 0
    all_special_tokens: list[str] = []

    def encode(self, text, add_special_tokens=False):
        return [1]


def _reference_layers(trunk: HFBackend, seq: list[int]) -> tuple:
    # An independent reference: the backend itself never uses output_hidden_states.
    with torch.inference_mode():
        out = trunk.model.model(input_ids=torch.tensor([seq]), output_hidden_states=True, return_dict=True)
    return out.hidden_states


def test_truncated_runs_match_output_hidden_states():
    trunk = _trunk()
    reference = _reference_layers(trunk, IDS)
    for k in range(1, FULL):
        taps, _ = trunk.prefill(IDS, None, (k,))
        assert _close(taps[k], reference[k][0, -1].float()), f"layer {k} differs"
    # Several depths from one forward, and a truncated prefill that a fork extends.
    taps, past = trunk.prefill(IDS[:6], None, (1, 3))
    extended, _ = trunk.prefill(IDS[6:], trunk.fork(past), (2, 3))
    assert _close(extended[2], reference[2][0, -1].float())
    assert _close(extended[3], reference[3][0, -1].float())


def test_trie_reads_at_mixed_depths_match_the_reference_layer():
    trunk = _trunk()
    depths = [2, 2, 1, 2, FULL, 1, 3]
    for overhead in (0, 64):
        reader = TrieReader(trunk, overhead_tokens=overhead)
        hiddens, record = reader.read(_branches(TRIE_SEQUENCES, depths))
        for i, (seq, depth) in enumerate(zip(TRIE_SEQUENCES, depths)):
            expected = _reference_layers(trunk, seq)[depth][0, -1].float()
            if depth == FULL:
                expected = _full(trunk, seq)
            assert _close(hiddens[f"b{i}"], expected), f"branch {i} at depth {depth} differs"
        # The stem is shared by four branches that need at most two layers.
        stem = [seg for seg in record["segments"] if seg["tokens"] == len(STEM)]
        if overhead == 0:
            assert stem and stem[0]["depth"] == 2
