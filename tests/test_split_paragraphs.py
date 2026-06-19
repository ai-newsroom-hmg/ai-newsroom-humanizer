"""Tests für split_paragraphs: robuste Paragraph-Aufteilung."""
from __future__ import annotations

from humanizer.core_bestofn import split_paragraphs


def test_double_newline_splits_when_paragraphs_large_enough():
    """Mit kleinen min/max-thresholds splittet auch bei \\n\\n."""
    # Beide Paragraphen gleich gross, damit sum > max_chars
    para = "Ein längerer journalistischer Absatz mit etwas Text für die Test-Schwelle. " * 8
    full = para + "\n\n" + para + "\n\n" + para
    chunks = split_paragraphs(full, min_chars=300, max_chars=800)
    assert len(chunks) >= 2, f"expected ≥2 chunks for 3 paragraphs of ~600 chars each, got {len(chunks)}"


def test_single_newline_fallback_demo_target():
    """Reproduziert das echte Demo-Target-Splitten (8515 chars, nur \\n, → 5 chunks)."""
    # 5 Absätze à ~1700 chars, getrennt durch \n (wie iCloud-Sync sie liefert)
    para = "Ein längerer journalistischer Absatz mit zahlreichen Sätzen und vielen Wörtern, der die Mindest-Länge des Splitters überschreitet. " * 13
    full = "\n".join([para, para, para, para, para])
    assert "\n\n" not in full  # nur single newlines im Input
    chunks = split_paragraphs(full, min_chars=400, max_chars=2000)
    assert len(chunks) >= 3, f"single-\\n fallback should yield ≥3 chunks for ~8500 chars, got {len(chunks)}"
    for c in chunks[:-1]:
        assert len(c) >= 400, f"chunk too small: {len(c)} chars"


def test_empty_text():
    chunks = split_paragraphs("")
    assert chunks == [] or all(not c.strip() for c in chunks)


def test_single_short_paragraph():
    chunks = split_paragraphs("Kurzer Satz.", min_chars=400)
    # zu kurz für eigenen chunk, kommt trotzdem als einer raus
    assert len(chunks) == 1
    assert chunks[0] == "Kurzer Satz."
