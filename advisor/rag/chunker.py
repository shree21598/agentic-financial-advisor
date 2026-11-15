"""Split markdown documents into heading-aware, overlapping chunks."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    section: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    out, step = [], max_words - overlap
    for i in range(0, len(words), step):
        out.append(" ".join(words[i : i + max_words]))
        if i + max_words >= len(words):
            break
    return out


def chunk_markdown(path: Path, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title_m = re.search(r"^#\s+(.+)$", raw, re.M)
    title = title_m.group(1).strip() if title_m else path.stem
    sections = re.split(r"^##\s+", raw, flags=re.M)
    chunks: list[Chunk] = []
    for si, sec in enumerate(sections):
        lines = sec.strip().splitlines()
        if not lines:
            continue
        if si == 0:
            name = "Overview"
            body = "\n".join(l for l in lines if not l.startswith("# "))
        else:
            name, body = lines[0].strip(), "\n".join(lines[1:])
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            continue
        for pi, piece in enumerate(_split_long(body, max_words, overlap)):
            chunks.append(
                Chunk(
                    id=f"{path.stem}#{si}.{pi}",
                    doc=path.name,
                    title=title,
                    section=name,
                    # Prefix with title/section so each chunk is self-describing for embeddings.
                    text=f"{title} - {name}: {piece}",
                )
            )
    return chunks


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for p in sorted(corpus_dir.glob("*.md")):
        out.extend(chunk_markdown(p))
    return out
