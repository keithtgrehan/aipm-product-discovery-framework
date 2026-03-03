from __future__ import annotations

from typing import List


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 150) -> List[str]:
    """Split text into overlapping character chunks."""
    clean = " ".join(text.split())
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    chunks: List[str] = []
    start = 0
    step = max(1, max_chars - overlap)

    while start < len(clean):
        end = min(start + max_chars, len(clean))
        chunk = clean[start:end]

        if end < len(clean):
            best_break = max(chunk.rfind(". "), chunk.rfind("; "), chunk.rfind(" "))
            if best_break > int(max_chars * 0.6):
                end = start + best_break + 1
                chunk = clean[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(clean):
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = start + step
        start = next_start

    return chunks
