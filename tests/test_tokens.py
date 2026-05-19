"""Tests for the Layout-Tokens prototype (RFC 0002).

Round-trip exactness on Manhattan synthetic batches is the headline
guarantee. Compression ratio + vocabulary size are verified at a sanity
level — RFC targets are aspirational for v0.2-GA.
"""

from __future__ import annotations

import torch

from openlithohub.synth import generate_synthetic_batch
from openlithohub.synth.rule_based import PatternKind
from openlithohub.tokens import (
    N_RESERVED,
    TOK_BOS,
    TOK_CLOSE,
    TOK_EOS,
    TOK_POLYGON,
    DecodedLayout,
    LayoutTokenizer,
)


def test_vocabulary_size() -> None:
    tok = LayoutTokenizer.from_pdk("freepdk45", canvas_size=256)
    assert tok.vocab_size == N_RESERVED + 257
    assert tok.coord_offset == N_RESERVED


def test_roundtrip_empty_mask() -> None:
    tok = LayoutTokenizer.from_pdk("freepdk45", canvas_size=64)
    mask = torch.zeros(64, 64)
    decoded = tok.decode(tok.encode(mask))
    assert isinstance(decoded, DecodedLayout)
    assert torch.equal(mask, decoded.mask)
    assert decoded.report.ok
    assert decoded.report.polygons_drawn == 0
    encoded = tok.encode(mask)
    assert encoded.ids.tolist() == [TOK_BOS, TOK_EOS]


def test_roundtrip_single_rectangle() -> None:
    tok = LayoutTokenizer.from_pdk("freepdk45", canvas_size=64)
    mask = torch.zeros(64, 64)
    mask[10:20, 5:25] = 1.0
    enc = tok.encode(mask)
    # Should be: BOS, POLYGON, 4 vertices (8 coord tokens), CLOSE, EOS
    ids = enc.ids.tolist()
    assert ids[0] == TOK_BOS
    assert ids[-1] == TOK_EOS
    assert ids.count(TOK_POLYGON) == 1
    assert ids.count(TOK_CLOSE) == 1
    decoded = tok.decode(enc)
    assert torch.equal(mask, decoded.mask)
    assert decoded.report.polygons_drawn == 1
    assert decoded.report.ok


def test_roundtrip_synthetic_sram() -> None:
    tok = LayoutTokenizer.from_pdk("freepdk45", canvas_size=256)
    batch = generate_synthetic_batch(PatternKind.SRAM, n=2, size=256, seed=1)
    for i in range(batch.masks.shape[0]):
        mask = batch.masks[i]
        decoded = tok.decode(tok.encode(mask))
        assert torch.equal(mask, decoded.mask), f"sample {i}: round-trip differs"
        assert decoded.report.ok


def test_roundtrip_synthetic_contact_array() -> None:
    tok = LayoutTokenizer.from_pdk("freepdk45", canvas_size=256)
    batch = generate_synthetic_batch(PatternKind.CONTACT_ARRAY, n=2, size=256, seed=2)
    for i in range(batch.masks.shape[0]):
        mask = batch.masks[i]
        decoded = tok.decode(tok.encode(mask))
        assert torch.equal(mask, decoded.mask), f"sample {i}: round-trip differs"
        assert decoded.report.ok


def test_compression_ratio_reasonable() -> None:
    """Tokens-per-pixel < 100% on a typical synthetic patch.

    The RFC target (<5%) is for v0.2-GA after Douglas-Peucker landed.
    The strip decomposition baseline can spend more tokens on ragged
    geometry; we only assert it is below the trivial pixel count.
    """
    tok = LayoutTokenizer.from_pdk("freepdk45", canvas_size=256)
    batch = generate_synthetic_batch(PatternKind.SRAM, n=1, size=256, seed=0)
    enc = tok.encode(batch.masks[0])
    pixels = 256 * 256
    assert len(enc.ids) < pixels


def test_vertices_are_in_canvas_range() -> None:
    tok = LayoutTokenizer.from_pdk("freepdk45", canvas_size=64)
    mask = torch.zeros(64, 64)
    mask[10:20, 5:25] = 1.0
    enc = tok.encode(mask)
    for t in enc.ids.tolist():
        if t >= N_RESERVED:
            v = tok._coord_value(t)
            assert 0 <= v <= 64


def test_decode_flags_unknown_token() -> None:
    """Out-of-vocabulary tokens between polygons must be reported, not eaten silently."""
    tok = LayoutTokenizer.from_pdk("freepdk45", canvas_size=64)
    bogus = tok.vocab_size + 7  # outside both reserved + coord ranges
    ids = torch.tensor([TOK_BOS, bogus, TOK_EOS], dtype=torch.int64)
    decoded = tok.decode(ids)
    assert decoded.report.unknown_tokens == 1
    assert decoded.report.polygons_drawn == 0
    assert not decoded.report.ok


def test_decode_flags_truncated_polygon() -> None:
    """A polygon opener with no CLOSE/EOS is reported as skipped + truncated."""
    tok = LayoutTokenizer.from_pdk("freepdk45", canvas_size=64)
    # POLYGON, then one stray coord pair, then end-of-stream — no CLOSE, no EOS.
    ids = torch.tensor(
        [TOK_BOS, TOK_POLYGON, tok._coord_id(0), tok._coord_id(0)],
        dtype=torch.int64,
    )
    decoded = tok.decode(ids)
    assert decoded.report.polygons_skipped == 1
    assert decoded.report.truncated
    assert decoded.report.polygons_drawn == 0
