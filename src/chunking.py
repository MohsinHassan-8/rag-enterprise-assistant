"""Recursive character text splitter with overlap."""

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_atoms(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split text into pieces of at most chunk_size, trying separators in order.

    Separators stay attached to the end of each piece, so joining the atoms
    reproduces the original text exactly.
    """
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # No separator left: hard split by characters
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_atoms(text, chunk_size, rest)

    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    atoms: list[str] = []
    for piece in pieces:
        if piece:
            atoms.extend(_split_atoms(piece, chunk_size, rest))
    return atoms


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text into overlapping chunks of at most chunk_size characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    atoms = _split_atoms(text, chunk_size, separators or DEFAULT_SEPARATORS)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for atom in atoms:
        if current and current_len + len(atom) > chunk_size:
            chunks.append("".join(current).strip())
            # Carry trailing atoms into the next chunk as overlap
            carried: list[str] = []
            carried_len = 0
            for a in reversed(current):
                if carried_len + len(a) > chunk_overlap:
                    break
                carried.insert(0, a)
                carried_len += len(a)
            current, current_len = carried, carried_len
            if current_len + len(atom) > chunk_size:
                current, current_len = [], 0
        current.append(atom)
        current_len += len(atom)
    if current:
        chunks.append("".join(current).strip())

    return [c for c in chunks if c]


def chunk_document(
    doc_id: str, text: str, title: str = "", url: str = "", **kwargs
) -> list[dict]:
    """Chunk one document and attach ids and metadata to every chunk."""
    return [
        {
            "chunk_id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "chunk_index": i,
            "title": title,
            "url": url,
            "text": chunk,
        }
        for i, chunk in enumerate(chunk_text(text, **kwargs))
    ]