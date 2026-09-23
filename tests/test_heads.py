import numpy as np
from fastapi.testclient import TestClient

from dynajev.compile import DecideIn
from dynajev.engine import Dynajev
from dynajev.heads import HeadParams, HeadStore, task_signature
from test_engine import KeywordBackend

TONE = {"tone": {"type": "choice", "instructions": "What is the tone?", "options": ["grateful", "angry"]}}
EXAMPLES = [
    {"context": f"{word} number {i}", "label": label}
    for i in range(4)
    for word, label in (("thanks", "grateful"), ("furious", "angry"))
]


def test_signature_ignores_spacing_and_case_but_not_labels_type_or_model():
    base = task_signature("choice", "What is the tone?", ["grateful", "angry"], "m")
    assert base == task_signature("choice", "  what is   the TONE? ", ["grateful", "angry"], "m")
    assert len(base) == 16
    assert base != task_signature("choice", "What is the tone?", ["angry", "grateful"], "m")
    assert base != task_signature("score", "What is the tone?", ["grateful", "angry"], "m")
    assert base != task_signature("choice", "What is the tone?", ["grateful", "angry"], "other-model")
    assert base != task_signature("choice", "What is the tone?", ["grateful", "angry"], "m", strategy="letter")


def _ridge_head() -> HeadParams:
    rng = np.random.default_rng(0)
    return HeadParams(
        chosen="ridge_probe",
        layer=12,
        weight=rng.normal(size=(2, 6)),
        ridge_bias=rng.normal(size=2),
        mu=rng.normal(size=6),
        sd=np.abs(rng.normal(size=6)) + 0.5,
        scale=4.0,
        signature="abc123",
        qtype="choice",
        labels=["grateful", "angry"],
        n_examples=8,
        metrics={"exit_layer": 12},
        instructions="What is the tone?",
    )


def test_store_round_trips_through_a_directory(tmp_path):
    head = _ridge_head()
    HeadStore(tmp_path).put(head)
    assert (tmp_path / "abc123.json").exists() and (tmp_path / "abc123.npz").exists()
    loaded = HeadStore(tmp_path).get("abc123")
    assert loaded is not None
    assert loaded.layer == 12 and loaded.scale == 4.0 and loaded.labels == ["grateful", "angry"]
    x = np.linspace(-1, 1, 6)
    assert np.allclose(loaded.ridge_logits(x), head.ridge_logits(x))
    listed = HeadStore(tmp_path).list()
    assert listed[0]["signature"] == "abc123" and listed[0]["exit_layer"] == 12
    affine = HeadParams(chosen="affine", temperature=1.5, bias=[0.2, -0.2], signature="def456", labels=["a", "b"])
    store = HeadStore(tmp_path)
    store.put(affine)
    assert not (tmp_path / "def456.npz").exists()
    again = HeadStore(tmp_path).get("def456")
    assert again is not None and np.allclose(again.affine_logits([1.0, 0.0]), affine.affine_logits([1.0, 0.0]))


def test_memory_store_evicts_the_oldest_head():
    store = HeadStore(capacity=2)
    for name in ("a", "b", "c"):
        store.put(HeadParams(chosen="affine", signature=name))
    assert store.get("a") is None and store.get("c") is not None


def test_saved_heads_apply_to_later_requests_without_examples():
    engine = Dynajev(KeywordBackend())
    fitted = engine.decide(
        DecideIn.model_validate({"context": "thanks a lot", "questions": TONE, "examples": EXAMPLES, "save_heads": True})
    )
    assert fitted["fit"]["saved"] is True
    signature = fitted["fit"]["signature"]
    later = engine.decide(DecideIn.model_validate({"context": "furious again", "questions": TONE}))
    field = later["fields"][0]
    assert field["head_source"] == "stored" and field["depth"] == 1
    assert later["answers"]["tone"]["choice"] == "angry"
    assert any(signature in note for note in later["notes"])
    assert signature in later["plan"]["fields"][0]["source"]
    plain = engine.decide(DecideIn.model_validate({"context": "furious again", "questions": TONE, "use_heads": False}))
    assert plain["fields"][0]["head_source"] == "readout" and plain["fields"][0]["depth"] is None
    # A reworded label set is a different task and does not pick up the head.
    other = {"tone": {**TONE["tone"], "options": ["angry", "grateful"]}}
    assert engine.decide(DecideIn.model_validate({"context": "x", "questions": other}))["fields"][0]["head_source"] == "readout"


def test_unsaved_fits_stay_out_of_the_store():
    engine = Dynajev(KeywordBackend())
    engine.decide(DecideIn.model_validate({"context": "thanks", "questions": TONE, "examples": EXAMPLES}))
    assert engine.heads.list() == []


def test_heads_api_fits_and_lists(tmp_path):
    from dynajev import server

    server._state["dynajev"] = Dynajev(KeywordBackend(), heads=HeadStore(tmp_path))
    try:
        client = TestClient(server.app)
        response = client.post("/api/heads/fit", json={"questions": TONE, "examples": EXAMPLES})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["heads"][0]["exit_layer"] == 1
        listed = client.get("/api/heads").json()
        assert listed["heads"][0]["signature"] == body["heads"][0]["signature"]
        assert listed["directory"] == str(tmp_path)
        bad = client.post("/api/heads/fit", json={"questions": {"q": {"type": "open", "instructions": "Why?"}}, "examples": EXAMPLES})
        assert bad.status_code == 400
    finally:
        server._state["dynajev"] = None
