import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from dynajev.trunk import Trunk

IDS = [7, 11, 3, 9, 4, 8, 2, 6, 5, 10]


def _tiny_qwen35() -> Qwen3_5ForCausalLM:
    # Qwen3.5 layout: linear-attention (Gated DeltaNet) layers with conv and
    # recurrent state, plus full-attention layers with K/V. The clone must carry both.
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
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


def test_branched_hidden_matches_a_full_forward_on_hybrid_deltanet():
    model = _tiny_qwen35()
    trunk = Trunk(model, _Stub(), model_id="tiny-qwen3.5")
    full = trunk._hidden(IDS, None)
    shared_hidden, past = trunk._forward(IDS[:6], None)
    cloned = trunk._clone_past(past)
    assert cloned is not None, "the cache clone fell back to a full forward"
    branched = trunk._hidden(IDS[6:], cloned)
    assert torch.allclose(full, branched, atol=1e-4, rtol=1e-4)
    # Three branches of different lengths, one batched forward. Each must match
    # its own full forward, and one of them is the shared prefix itself.
    others = [IDS[:6] + [1, 2, 3], IDS[:6] + [12, 13, 14, 15, 16, 17, 18], IDS[:6]]
    via_read, shared = trunk.read([IDS, *others])
    assert shared == 6
    assert len(via_read) == 4
    assert torch.allclose(via_read[0], full, atol=1e-4, rtol=1e-4)
    for hidden, seq in zip(via_read[1:], others):
        assert torch.allclose(hidden, trunk._hidden(seq, None), atol=1e-4, rtol=1e-4)
    assert trunk._branch_batch(past, [IDS[6:], [1, 2, 3]]) is not None, "batched branch fell back"
    # The original cache must be untouched by the branch, so a second branch is clean.
    again = trunk._hidden(IDS[6:], trunk._clone_past(past))
    assert torch.allclose(full, again, atol=1e-4, rtol=1e-4)
    assert shared_hidden.shape[-1] == model.config.hidden_size


def test_prefix_cache_reuses_the_state_prefill_across_requests():
    trunk = Trunk(_tiny_qwen35(), _Stub(), model_id="tiny-qwen3.5")
    trunk.prefix_cache_size = 2
    first, shared = trunk.read([IDS], anchor=6)
    assert shared == 6 and trunk.last_read_cached is False
    assert trunk.prefix_cache_stats()["misses"] == 1
    other = IDS[:6] + [20, 21]
    second, shared = trunk.read([other, IDS], anchor=6)
    assert shared == 6 and trunk.last_read_cached is True
    assert trunk.prefix_cache_stats()["hits"] == 1
    assert torch.allclose(second[0], trunk._hidden(other, None), atol=1e-4, rtol=1e-4)
    assert torch.allclose(second[1], first[0], atol=1e-4, rtol=1e-4)
    # Eviction keeps the newest entries.
    trunk.read([[1, 2, 3, 4, 5]], anchor=3)
    trunk.read([[9, 9, 9, 9]], anchor=2)
    assert trunk.prefix_cache_stats()["entries"] == 2


def test_dense_read_matches_full_forwards_row_by_row():
    trunk = Trunk(_tiny_qwen35(), _Stub(), model_id="tiny-qwen3.5")
    rows = [IDS, IDS[:4], [3, 1, 4, 1, 5, 9, 2, 6], [7]]
    dense = trunk.read_dense(rows, chunk_rows=3)
    assert len(dense) == 4
    for hidden, seq in zip(dense, rows):
        assert torch.allclose(hidden, trunk._hidden(seq, None), atol=1e-4, rtol=1e-4)


def test_unknown_layout_is_refused():
    class Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"hidden_size": 4, "vocab_size": 8})()

    try:
        Trunk(Bare(), _Stub(), model_id="bare")
    except ValueError as exc:
        assert "model.model" in str(exc)
    else:
        raise AssertionError("a model without the expected layout was accepted")


class _Stub:
    eos_token_id = 0
    all_special_tokens: list[str] = []

    def encode(self, text, add_special_tokens=False):
        return [1]
