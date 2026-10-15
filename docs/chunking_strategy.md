# Chunking strategy

## Why chunk

The embedding model (all-MiniLM-L6-v2) reads at most 256 tokens and silently truncates anything longer. A single vector for a whole article would also blur many topics together. Splitting articles into passages lets retrieval return the specific passage that answers a question and keeps the context passed to the LLM small.

## Strategy: recursive character splitting with overlap

Implemented in `src/chunking.py` with no external dependency and covered by 17 unit tests in `tests/test_chunking.py`.

Parameters: `chunk_size` = 800 characters, `chunk_overlap` = 100 characters. Separators in priority order: paragraph break, line break, sentence end (". "), space. If a piece still exceeds `chunk_size`, it is hard-split by characters.

Algorithm:

1. Split the text at the highest-priority separator. Any piece longer than `chunk_size` is split again with the next separator. The separator stays attached to the end of each piece, so no text is lost.
2. Pack the pieces in order into a chunk until the next piece would exceed `chunk_size`.
3. Start the next chunk with the last whole pieces of the previous chunk, up to `chunk_overlap` characters.
4. Give each chunk the id `<doc_id>_<chunk_index>` plus metadata (doc_id, chunk_index, title, url).

## Why these values

- 800 characters: chunks have a median of 146 tokens (max 522), under the model's 256-token limit for 99.56% of chunks.
- 100 characters of overlap (12.5%): keeps context for sentences near a boundary without inflating the index much.
- Paragraph-first splitting keeps related sentences together where the article structure allows it.

## Measured results

Input: 6,960 cleaned documents.

| Metric | Value |
|--------|-------|
| Chunks before filtering | 34,556 |
| Dropped (under 50 characters) | 47 |
| Chunks embedded and indexed | 34,509 |
| Chunks per document | median 2, mean 5.0, max 85 |
| Characters per chunk | min 50, p5 294, median 674, mean 621, max 800 |
| Chunks under 100 characters | 104 (0.3%) |
| Tokens per chunk | median 146, max 522 |
| Chunks over the 256-token limit | 151 (0.44%) |

## Edge cases handled

- Empty or whitespace-only text returns no chunks.
- Text shorter than `chunk_size` returns a single chunk.
- Text with no usable separator (one long unbroken string) is hard-split by characters.
- Invalid parameters (`chunk_size` <= 0, overlap < 0 or >= `chunk_size`) raise `ValueError`.
- Output is deterministic: the same input always gives the same chunks and ids.
- Chunks under 50 characters are dropped at build time, because fragments such as stray headings make weak retrieval targets.

## Known limitations

- 151 chunks (0.44%) exceed 256 tokens. Their vectors represent only the first 256 tokens, while the full text is still stored and returned.
- Overlap is made of whole pieces, so the actual overlap is between 0 and 100 characters. If the final piece of a chunk is longer than 100 characters, the next chunk has no overlap.
- Sizes are measured in characters, which only approximates tokens. A token-based splitter would be more exact and was not implemented.
- List-style pages (for example a year page with one event per line) are split into groups of lines, and a chunk may not say which year it belongs to. The title is stored as metadata but is not part of the embedded text.
- Only one configuration (800/100) was built and benchmarked. Chunk size was not tuned against retrieval quality.
