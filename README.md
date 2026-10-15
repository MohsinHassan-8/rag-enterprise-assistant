# RAG Enterprise Assistant

End-to-end Retrieval-Augmented Generation (RAG) pipeline: data ingestion, embedding, vector search, grounded LLM answers, NLP analysis, and a FastAPI service. Built week by week; this README is updated with each submission.

## Progress

| Week | Focus | Status |
|------|-------|--------|
| 1 | Environment, data ingestion, preprocessing | Done |
| 2 | Chunking, embeddings, vector database, retrieval benchmark | Done |

## Setup

Requires Python 3.12 and an NVIDIA GPU with a current driver. Developed on Windows with a GTX 1650 (4 GB).

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install torch first. Otherwise pip's resolver can backtrack to ancient `sentence-transformers` versions and fail on Python 3.12.

## Week 1: Environment, Data Ingestion & Preprocessing

### Delivered

- Git repo with venv, `.gitignore` (data, `.env`, vector DB), and pinned `requirements.txt`
- `scripts/verify_env.py`: checks CUDA availability and core imports (pandas, sentence-transformers, ChromaDB, spaCy)
- Dataset: 7,000 shuffled Simple English Wikipedia articles (`wikimedia/wikipedia`, config `20231101.simple`) saved to `data/raw/`
- `src/preprocessing.py`: HTML stripping, Unicode normalization, whitespace cleanup, language filtering, and removal of trailing link-list sections (References, Other websites)
- `tests/test_preprocessing.py`: 36 unit tests
- `data/processed/clean_corpus.parquet`: validated clean corpus

### Corpus build results

| Stage | Documents |
|-------|-----------|
| Raw | 7,000 |
| Dropped: under 200 chars after cleaning | 26 |
| Dropped: non-English | 14 |
| Dropped: duplicates | 0 |
| **Final** | **6,960** (about 3.46M words, median 1,347 chars) |

### Approach and design decisions

- **Dataset:** Simple English Wikipedia was chosen over ArXiv or Reddit because it needs no authentication, gives clean article-level documents, and the articles are long enough to chunk properly. It is streamed from Hugging Face with a fixed shuffle seed (42) and a 10,000-article shuffle buffer. The sample is reproducible but not uniformly random, because buffered shuffling draws from roughly the first 20,000 articles of the dump. 7,000 documents were downloaded to leave margin above 5,000 after filtering.
- **Cleaning order:** HTML stripping, Unicode normalization, whitespace cleanup, then trailing-section removal. Each step prepares the next: block tags become newlines, non-breaking spaces become plain spaces, and the heading match needs trimmed lines.
- **HTML stripping:** BeautifulSoup is only invoked when a tag-like pattern is present, so plain-text comparisons such as "5 < 7" are preserved. Script and style contents are dropped, and entities are decoded.
- **Unicode:** NFKC normalization, zero-width and control characters removed, curly quotes straightened, so equivalent text matches consistently at retrieval time.
- **Language filtering:** langdetect on the first 1,000 characters, seeded for determinism. It is kept separate from `clean_document()` because it is a keep/drop decision, not a text transformation.
- **Trailing sections (beyond the brief):** "References" and "Other websites" sections are link lists that add no retrievable content, so they are cut.
- **Corpus build:** documents under 200 characters after cleaning are dropped, then non-English documents, then duplicates. Assertions run before the file is written, so an invalid corpus cannot be saved silently.

### Limitations

- langdetect can misclassify short or name-heavy articles. The 14 documents dropped as non-English were not manually reviewed.
- Language is detected from the first 1,000 characters only.
- NFKC normalization changes some characters (ligatures, full-width forms). This is intentional for retrieval consistency.

### Reproduce Week 1

```powershell
python scripts\verify_env.py        # environment check
python scripts\download_data.py     # raw data -> data/raw/ (git-ignored)
python -m pytest -q                 # unit tests
python scripts\build_corpus.py      # -> data/processed/clean_corpus.parquet
```

`scripts/inspect_raw.py` prints dataset statistics and noise indicators for the raw data.

## Week 2: Chunking, Embeddings & Vector DB

### Delivered

- `src/chunking.py`: recursive character splitter with overlap (17 unit tests). Strategy documented in `docs/chunking_strategy.md`
- `scripts/build_chunks.py`: corpus to chunks, with validation
- `scripts/embed_chunks.py`: sentence-transformers embeddings on the GPU, timing log in `reports/embedding_log.json`
- `src/vectorstore.py`: ChromaDB wrapper (persistent store, cosine distance, validated batched upsert, search) with 24 edge-case unit tests
- `scripts/ingest_chroma.py`: ingests the chunks and runs four checks (count, idempotent re-upsert, vector and text round-trip, distance equals cosine). Log in `reports/ingest_log.json`
- `scripts/search_demo.py`: command-line semantic search
- `scripts/benchmark_retrieval.py`: latency benchmark, results in `reports/retrieval_benchmark.json`
- Total test suite: 77 tests

### Measured results

Hardware: NVIDIA GTX 1650 (4 GB). Single run on one machine, so absolute numbers are not portable.

| Stage | Result |
|-------|--------|
| Chunks built | 34,556, of which 47 under 50 chars were dropped, leaving 34,509 |
| Chunk length | median 674 chars (max 800), median 146 tokens |
| Embedding model | all-MiniLM-L6-v2, 384 dimensions, 256-token limit |
| Embedding time | 132.8 s for 34,509 chunks (260 chunks/s, 3.85 ms per chunk), peak GPU memory 0.41 GB |
| ChromaDB ingest | 33.2 s, database size 226 MB |

Retrieval latency, 20 queries x 20 repetitions, k=5, 34,509 chunks:

| Component | p50 (ms) | p95 (ms) |
|-----------|----------|----------|
| Query embedding | 8.25 | 11.20 |
| Chroma search | 5.18 | 6.28 |
| **Total** | **13.32** | **16.79** (p99 18.47) |

- The first query after startup took 200.9 ms (one-time initialization).
- Search latency stayed between 4.9 and 6.1 ms (p50) as k grew from 1 to 50. Query type (question, keyword, long, numeric, nonsense) made no meaningful difference.
- A metadata-filtered search took 27.8 ms (p50) against 4.6 ms for the same unfiltered query. The cause was not investigated.
- Top-1 similarity scores: nonsense query 0.363, out-of-domain query 0.315, lowest in-domain query 0.481.

### Approach and design decisions

- **Chunking:** a self-written recursive splitter (paragraph, line, sentence, word, then character) with 800-character chunks and 100 characters of overlap, chosen so chunks stay under the model's 256-token limit. See `docs/chunking_strategy.md`.
- **Embeddings:** all-MiniLM-L6-v2 is small and fast and fits comfortably on a 4 GB GPU. Vectors are normalized and the collection uses cosine distance.
- **Benchmarking method:** the model is warmed up before timing, GPU work is synchronized before reading the clock, and embedding time and search time are measured separately.
- **Idempotent ingestion:** chunk ids are `<doc_id>_<chunk_index>` and ingestion uses upsert, so re-running does not create duplicates. `--reset` rebuilds from scratch.
- **Edge cases handled in `src/vectorstore.py`:** mismatched lengths, duplicate ids, NaN or infinite vectors, non-2-D arrays, empty documents (skipped and counted), empty collection (returns an empty list), `k` larger than the collection (clamped), invalid `k`, invalid query vectors, and filters that match nothing.
- **What is committed:** timing and benchmark logs in `reports/`. Embeddings, chunk files, and `chroma_db/` are git-ignored because they are large and regenerable.

### Limitations

- 151 chunks (0.44%) exceed the model's 256-token limit, so their vectors represent only the first 256 tokens. The full text is still stored and returned.
- The relevance check in the benchmark is a smoke test on 5 queries whose expected articles were chosen after viewing demo output. It is not an evaluation. Precision and recall are not measured yet.
- Score observations come from 20 queries, so no similarity threshold is validated. The corpus is a 7,000-article sample, so some topics are missing: "Mozart" returned the "Austria" article.
- Only one chunking configuration (800/100) was built and benchmarked.
- The article title is stored as metadata but is not part of the embedded text.

### Reproduce Week 2

```powershell
python scripts\build_chunks.py              # -> data/processed/chunks.parquet
python scripts\embed_chunks.py              # downloads the model on first run
python scripts\ingest_chroma.py --reset     # -> chroma_db/
python scripts\search_demo.py "How do solar eclipses happen?" --k 3
python scripts\benchmark_retrieval.py       # -> reports/retrieval_benchmark.json
python -m pytest -q                         # 77 tests
```

## Project structure

```
data/raw/            raw downloads (git-ignored)
data/processed/      corpus, chunks, embeddings (git-ignored)
chroma_db/           persistent vector database (git-ignored)
docs/                chunking_strategy.md
reports/             embedding, ingest, and retrieval benchmark logs
scripts/             verify_env, download_data, inspect_raw, build_corpus,
                     build_chunks, embed_chunks, ingest_chroma, search_demo,
                     benchmark_retrieval
src/                 preprocessing.py, chunking.py, vectorstore.py
tests/               pytest suite
```

## Data source

Articles come from the `wikimedia/wikipedia` dataset on Hugging Face (config `20231101.simple`). Wikipedia text is available under CC BY-SA and GFDL licenses; see the dataset card for details.

## Configuration

Copy `.env.example` to `.env` for API keys (needed from Week 3). `.env` is git-ignored.
