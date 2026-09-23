"""Read many branches by prefilling each shared token segment once.

The branch token sequences of a request form a trie. Every segment shared by
two or more branches can be prefilled once into a cache, and the branches
below it fork that cache; the rest of the branches run as one right-padded
batch off their parent's cache. The result is exact: every hidden state equals
the one a full forward of that branch would give.

Whether a shared segment gets its own forward is a cost decision. A forward
has a fixed cost (on the measured CPU about 100 ms, the time to stream the
weights) on top of a per-token cost (about 1.5 ms). Splitting a segment of
length L shared by n rows saves (n - 1) * L tokens and costs one more
forward, so it is split only when the saving exceeds `overhead_tokens`.
With `overhead_tokens=0` every shared segment is split.

Each branch may ask for its own depth (decoder layers to run). A trie node
runs to the deepest depth any branch below it needs; shallower branches that
end at that node tap the intermediate layer.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Node:
    tokens: list[int]
    children: list["_Node"] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    rows: int = 0
    depth: int = 0
    kept: bool = False


def build_trie(sequences: list[list[int]], depths: list[int], kept: list[bool] | None = None) -> _Node:
    kept = kept or [False] * len(sequences)
    return _build(list(range(len(sequences))), sequences, depths, kept, 0)


def _build(index: list[int], sequences: list[list[int]], depths: list[int], kept: list[bool], offset: int) -> _Node:
    end = offset
    while True:
        if any(len(sequences[i]) <= end for i in index):
            break
        token = sequences[index[0]][end]
        if any(sequences[i][end] != token for i in index):
            break
        end += 1
    node = _Node(tokens=list(sequences[index[0]][offset:end]))
    groups: OrderedDict[int, list[int]] = OrderedDict()
    for i in index:
        if len(sequences[i]) == end:
            node.ends.append(i)
        else:
            groups.setdefault(sequences[i][end], []).append(i)
    node.children = [_build(group, sequences, depths, kept, end) for group in groups.values()]
    node.rows = len(index)
    node.depth = max(depths[i] for i in index)
    node.kept = any(kept[i] for i in node.ends)
    return node


class PrefixStore:
    """Prefilled state prefixes kept across requests, keyed by their exact tokens."""

    def __init__(self, capacity: int = 0):
        self.capacity = capacity
        self._entries: OrderedDict[tuple[int, ...], tuple[Any, int]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[int, ...], depth: int) -> Any:
        entry = self._entries.get(key)
        if entry is None or entry[1] < depth:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key: tuple[int, ...], cache: Any, depth: int) -> None:
        if self.capacity <= 0:
            return
        self._entries[key] = (cache, depth)
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._entries), "capacity": self.capacity, "hits": self.hits, "misses": self.misses}


class TrieReader:
    def __init__(self, backend: Any, prefix_cache: int = 0, overhead_tokens: int = 64, chunk_rows: int = 64):
        self.backend = backend
        self.store = PrefixStore(prefix_cache)
        self.overhead_tokens = overhead_tokens
        self.chunk_rows = chunk_rows

    @property
    def full_depth(self) -> int:
        return int(getattr(self.backend, "num_layers", 0) or 1)

    def read(
        self,
        branches: list[Any],
        anchor: list[int] | None = None,
        kept: dict[tuple[int, ...], tuple[dict[int, Any], Any, int]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Hidden state at the last position of every branch, at the branch's depth.

        `anchor` is the token prefix of the state block; its prefill is kept in
        the cross-request prefix store. `kept` holds exact caches of earlier
        branches in the same request (for questions that continue another
        question's conversation); branches that extend one of them start there,
        and branches marked `keep` are added to it.
        """

        full = self.full_depth
        sequences = [list(b.tokens or []) for b in branches]
        depths = [min(full, b.depth or full) for b in branches]
        keep = [bool(getattr(b, "keep", False)) for b in branches]
        run = _Run(self, sequences, depths, keep, kept if kept is not None else {})
        groups: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
        for i, seq in enumerate(sequences):
            groups.setdefault(run.longest_kept(seq, depths[i]), []).append(i)
        for key, index in groups.items():
            run.group(index, key, anchor if not key else None)
        hiddens = {b.id: run.out[i] for i, b in enumerate(branches)}
        naive = sum(len(s) for s in sequences)
        record = {
            "branches": len(branches),
            "naive_tokens": naive,
            "processed_tokens": run.processed,
            "padded_tokens": run.padded,
            "cached_tokens": run.cached,
            "prefix_cache_hit_tokens": run.store_hit,
            "forwards": run.forwards,
            "shared_prefix_tokens": run.shared_prefix,
            "segments": run.segments,
            "batches": run.batches,
        }
        return hiddens, record


class _Run:
    def __init__(self, reader: TrieReader, sequences, depths, keep, kept):
        self.reader = reader
        self.backend = reader.backend
        self.sequences = sequences
        self.depths = depths
        self.keep = keep
        self.kept = kept
        self.out: dict[int, Any] = {}
        self.processed = 0
        self.padded = 0
        self.cached = 0
        self.store_hit = 0
        self.forwards = 0
        self.shared_prefix = 0
        self.segments: list[dict[str, Any]] = []
        self.batches: list[dict[str, Any]] = []

    def longest_kept(self, seq: list[int], depth: int) -> tuple[int, ...]:
        best: tuple[int, ...] = ()
        for key, (_, _, kept_depth) in self.kept.items():
            if len(best) < len(key) <= len(seq) and kept_depth >= depth and tuple(seq[: len(key)]) == key:
                best = key
        return best

    def group(self, index: list[int], key: tuple[int, ...], anchor: list[int] | None) -> None:
        offset = len(key)
        cache = None
        if key:
            taps, cache, _ = self.kept[key]
            self.cached += offset
            for i in index:
                if len(self.sequences[i]) == offset:
                    self.out[i] = taps.get(self.depths[i], taps[max(taps)])
            index = [i for i in index if len(self.sequences[i]) > offset]
            if not index:
                return
        root = build_trie([self.sequences[i][offset:] for i in index], [self.depths[i] for i in index], [self.keep[i] for i in index])
        _remap(root, index)
        if root.rows > 1:
            self.shared_prefix = max(self.shared_prefix, offset + len(root.tokens))
        store = self.reader.store
        if anchor is not None and store.capacity > 0:
            limit = len(root.tokens) - (1 if root.ends else 0)
            cut = 0
            while cut < min(len(anchor), limit) and anchor[cut] == root.tokens[cut]:
                cut += 1
            if cut > 0:
                prefix = tuple(root.tokens[:cut])
                cache = store.get(prefix, root.depth)
                if cache is not None:
                    store.hits += 1
                    self.cached += cut
                    self.store_hit += cut
                    self.segments.append({"tokens": cut, "rows": root.rows, "depth": root.depth, "cached": True})
                else:
                    store.misses += 1
                    _, cache = self.backend.prefill(list(prefix), None, (root.depth,))
                    self._count(cut, cut)
                    self.segments.append({"tokens": cut, "rows": root.rows, "depth": root.depth})
                    store.put(prefix, cache, root.depth)
                root.tokens = root.tokens[cut:]
        if not root.tokens:
            self.children(root.children, cache, offset)
        elif root.kept or (root.rows > 1 and self._worth(root)):
            self.node(root, cache, offset)
        else:
            self.children([root], cache, offset)

    def _worth(self, node: _Node) -> bool:
        return (node.rows - 1) * len(node.tokens) >= self.reader.overhead_tokens

    def _count(self, real: int, padded: int) -> None:
        self.processed += real
        self.padded += padded
        self.forwards += 1

    def node(self, node: _Node, cache: Any, offset: int) -> None:
        layers = tuple(sorted({node.depth, *(self.depths[i] for i in node.ends)}))
        taps, cache = self.backend.prefill(node.tokens, self.backend.fork(cache), layers)
        self._count(len(node.tokens), len(node.tokens))
        self.segments.append({"tokens": len(node.tokens), "rows": node.rows, "depth": node.depth})
        for i in node.ends:
            self.out[i] = taps[self.depths[i]]
            if self.keep[i]:
                self.kept[tuple(self.sequences[i])] = (taps, cache, node.depth)
        self.children(node.children, cache, offset + len(node.tokens))

    def children(self, children: list[_Node], cache: Any, offset: int) -> None:
        rows: list[tuple[int, list[int]]] = []
        queue = list(children)
        while queue:
            child = queue.pop(0)
            if child.rows == 1 and not child.children and not child.kept:
                rows.extend((i, child.tokens) for i in child.ends)
            elif child.kept or (child.rows > 1 and self._worth(child)):
                self.node(child, cache, offset)
            else:
                queue.extend(
                    _Node(tokens=child.tokens + c.tokens, children=c.children, ends=c.ends, rows=c.rows, depth=c.depth, kept=c.kept)
                    for c in child.children
                )
                rows.extend((i, child.tokens) for i in child.ends)
        if not rows:
            return
        suffixes = [tokens for _, tokens in rows]
        layers = [(self.depths[i],) for i, _ in rows]
        chunk = self.reader.chunk_rows
        results = self.backend.prefill_rows(cache, suffixes, layers, chunk)
        for (i, _), taps in zip(rows, results):
            self.out[i] = taps[self.depths[i]]
        for start in range(0, len(suffixes), chunk):
            part = suffixes[start : start + chunk]
            width = max(len(s) for s in part)
            self._count(sum(len(s) for s in part), width * len(part))
            self.batches.append({"rows": len(part), "width": width})


def _remap(node: _Node, index: list[int]) -> None:
    node.ends = [index[i] for i in node.ends]
    for child in node.children:
        _remap(child, index)
