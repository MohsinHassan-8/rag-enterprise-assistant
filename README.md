# RAG Enterprise Assistant

End-to-end Retrieval-Augmented Generation (RAG) pipeline: data ingestion, embedding, vector search, grounded LLM answers, NLP analysis, and a FastAPI service. Built week by week; this README is updated with each submission.

## Progress

| Week | Focus | Status |
|------|-------|--------|
| 1 | Environment, data ingestion, preprocessing | Done |

## Week 1: Environment, Data Ingestion & Preprocessing

**Delivered**
- Git repo with venv, `.gitignore` (data, `.env`, vector DB), and pinned `requirements.txt`
- `scripts/verify_env.py`: asserts CUDA availability and core imports (pandas, sentence-transformers, ChromaDB, spaCy)
- Dataset: 7,000 random Simple English Wikipedia articles (`wikimedia/wikipedia`, config `20231101.simple`) saved to `data/raw/`
- `src/preprocessing.py`: HTML stripping, Unicode normalization, whitespace cleanup, language filtering, and removal of trailing link-list sections (References, Other websites)
- `tests/test_preprocessing.py`: 36 unit tests
- `data/processed/clean_corpus.parquet`: validated clean corpus

**Corpus build results**

| Stage | Documents |
|-------|-----------|
| Raw | 7,000 |
| Dropped: under 200 chars after cleaning | 26 |
| Dropped: non-English | 14 |
| Dropped: duplicates | 0 |
| **Final** | **6,960** (about 3.46M words, median 1,347 chars) |

## Setup

Requires Python 3.12 and, for GPU use, an NVIDIA GPU with a current driver. Developed on Windows with a GTX 1650.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install torch first. Otherwise pip's resolver can backtrack to ancient `sentence-transformers` versions and fail on Python 3.12.

## Reproduce Week 1

```powershell
python scripts\verify_env.py        # environment check
python scripts\download_data.py     # raw data -> data/raw/ (git-ignored)
python -m pytest -q                 # unit tests
python scripts\build_corpus.py      # -> data/processed/clean_corpus.parquet
```

`scripts/inspect_raw.py` prints dataset statistics and noise indicators for the raw data.

## Project structure

```
data/raw/            raw downloads (git-ignored)
data/processed/      clean corpus (git-ignored)
scripts/             verify_env, download_data, inspect_raw, build_corpus
src/preprocessing.py cleaning functions
tests/               pytest suite
```

## Configuration

Copy `.env.example` to `.env` for API keys (needed from Week 3). `.env` is git-ignored.

## Approach and design decisions

- **Dataset:** Simple English Wikipedia was chosen over ArXiv or Reddit because it needs no authentication, gives clean article-level documents, and the articles are long enough to chunk properly in Week 2. It is streamed from Hugging Face with a fixed shuffle seed (42) for a reproducible random sample. 7,000 documents were downloaded to leave margin above 5,000 after filtering.
- **Cleaning order:** HTML stripping, Unicode normalization, whitespace cleanup, then trailing-section removal. Whitespace cleanup runs before section removal because the heading match depends on trimmed lines, and earlier steps can introduce new whitespace (block tags become newlines, non-breaking spaces become spaces).
- **HTML stripping:** BeautifulSoup is only invoked when a tag-like pattern is present, so plain-text comparisons such as "5 < 7" are preserved. Script and style contents are dropped, and entities are decoded.
- **Unicode:** NFKC normalization, zero-width and control characters removed, curly quotes straightened, so equivalent text matches consistently at retrieval time.
- **Language filtering:** langdetect on the first 1,000 characters, seeded for determinism. It is kept separate from `clean_document()` because it is a keep/drop decision, not a text transformation, and is applied in `scripts/build_corpus.py`.
- **Trailing sections (beyond the brief):** "References" and "Other websites" sections are link lists that add no retrievable content and would pollute retrieval in Week 2, so they are cut.
- **Corpus build:** documents under 200 characters after cleaning are dropped, then non-English documents, then duplicates. Hard assertions run before the file is written (no HTML tags, no tabs, no double spaces, no blank-line runs, unique ids, minimum length), so an invalid corpus cannot be saved silently.
- **Testing:** 36 unit tests cover each cleaner, including edge cases and an idempotency check on the full pipeline.

## Limitations

- langdetect can misclassify short or name-heavy articles. The 14 documents dropped as non-English were not manually reviewed.
- Language is detected from the first 1,000 characters only.
- NFKC normalization changes some characters (ligatures, full-width forms). This is intentional for retrieval consistency.

## Data source

Articles come from the `wikimedia/wikipedia` dataset on Hugging Face (config `20231101.simple`). Wikipedia text is available under CC BY-SA and GFDL licenses; see the dataset card for details.
