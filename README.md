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
