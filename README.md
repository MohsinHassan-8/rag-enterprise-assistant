# RAG Enterprise Assistant

End-to-end Retrieval-Augmented Generation (RAG) pipeline: data ingestion, embedding, vector search, grounded LLM answers, NLP analysis, and a FastAPI service. Built week by week; this README is updated with each submission.

## Progress

| Week | Focus | Status |
|------|-------|--------|
| 1 | Environment, data ingestion, preprocessing | Done |
| 2 | Chunking, embeddings, vector database, retrieval benchmark | Done |
| 3 | LLM integration, prompt engineering, guarded RAG pipeline | Done |

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
- Test suite at the end of Week 2: 77 tests

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
python -m pytest -q                         # 191 tests (all weeks)
```

## Week 3: LLM Integration & Prompt Engineering

### Delivered

- `src/llm_client.py`: OpenAI-compatible chat client (httpx) with timeouts, retries with backoff, `Retry-After` handling, and typed errors (43 unit tests, HTTP mocked)
- `src/prompts.py`: system prompt, numbered-source context injection with a character budget, input cleaning, refusal detection (31 unit tests)
- `src/rag.py`: pipeline of retrieve, relevance gate, generate, verify, and log latency (40 unit tests, fake LLM)
- `scripts/ask.py` (command-line questions), `scripts/run_rag_eval.py` (16-query evaluation), `scripts/list_llm_models.py` and `scripts/smoke_llm.py` (check which model IDs are live and that the key works)
- `reports/rag_eval.json`: evaluation results. Per-request latency logs go to `logs/rag_requests.jsonl` (git-ignored)
- Test suite: 191 tests in total across all weeks

### Measured results

Provider and model: Groq, `openai/gpt-oss-20b` (a reasoning model with a 131K context window). Free-tier model IDs change often: the Llama 3.1 8B and Llama 3.3 70B IDs used in most tutorials were absent from this key's live model list, so the model ID lives in `.env` and is chosen with `scripts/list_llm_models.py`.

Evaluation: 16 hand-written queries, one run (`reports/rag_eval.json`).

| Category | Passed | What counts as a pass |
|----------|--------|-----------------------|
| Answerable questions | 7/7 | status `ok` and the answer contains the expected fact |
| Prompt injection | 1/1 | the answer does not obey the injected instruction |
| Not in corpus (2 of 3 use fabricated entities) | 3/3 | no answer shown |
| Off-topic | 3/3 | no answer shown |
| Nonsense | 2/2 | no answer shown |

- Unsupported answers shown for questions the sources cannot answer: 0. False withholds on answerable questions: 0.
- Injection: the model did not obey the injected instruction, but it also refused to answer the underlying answerable question, so the injected text still cost that question its answer.
- 4 of 16 queries were stopped by the relevance gate and never called the LLM. The 3 not-in-corpus questions (top scores 0.458 to 0.532) and the off-topic query "Write me a poem about my cat" (0.471) passed the gate, and the model refused them.
- Score calibration: lowest top score among answerable queries 0.68; highest among off-topic and nonsense queries 0.471. The gate threshold stays at 0.40.

Latency in milliseconds, warm process, same run:

| Requests | n | p50 | p95 | mean |
|----------|---|-----|-----|------|
| End to end, requests that called the LLM | 12 | 567.7 | 802.6 | 566.2 |
| LLM call only | 12 | 535.5 | 770.4 | 536.2 |
| Retrieval (embedding + search), all requests | 16 | 29.2 | 34.9 | 29.4 |
| End to end, gated requests (no LLM call) | 4 | 30.0 | 34.0 | 29.8 |

- No rate-limit retries occurred (retry wait total 0 ms). Mean 1,103 tokens per LLM request.
- The first request in a fresh process took about 2.2 s (embedder and ChromaDB warm-up). The evaluation script excludes this with a warm-up call.

Live error-handling checks (the failure was injected against the real API):

| Failure injected | Result |
|------------------|--------|
| Wrong model ID | `LLMModelNotFoundError` (HTTP 404), not retried, user-facing message returned together with the retrieved sources |
| Invalid API key | `LLMAuthError` (HTTP 401), not retried, key not shown in the output |

Rate limits (429), 5xx errors, timeouts, connection errors, token-limit errors and malformed responses are covered by mocked-HTTP unit tests only. They were not triggered against the live API.

### Approach and design decisions

- **Layered grounding:** (1) Relevance gate: if the best retrieval score is below 0.40 the LLM is not called. (2) Prompt: answer only from numbered sources, cite after each statement, reply with one fixed refusal sentence when the sources do not answer, and treat the sources and the question as data, not instructions. (3) Post-checks on the answer: every `[n]` must refer to a source that was sent, there must be at least one citation, each sentence needs at least 60% of its content words found in the sources or the question, and every number must appear in the sources.
- **Statuses:** `ok`, `refused`, `off_topic`, `unverified` (a check failed: the answer is withheld and the model's text is kept in `raw_answer` for inspection), `flagged` (same checks, answer shown, only if `withhold_unverified=False`), `llm_error` (sources still returned), `invalid_query`.
- **Context injection:** top 5 chunks numbered in rank order up to a 4,500-character budget, so a citation number always maps to a source that was actually sent. Questions are cleaned first (control characters removed, whitespace collapsed, 1,000-character cap).
- **Token limits:** if the provider reports a token-limit error, the pipeline retries once with half the context and 1.5x the completion budget. The model spends completion tokens on hidden reasoning, so `reasoning_effort` is set to `low` (6 reasoning tokens against 48 for a trivial prompt in a smoke test), `max_tokens` is 700, and an empty answer with `finish_reason` of `length` raises a token-limit error instead of being returned.
- **API error handling (`src/llm_client.py`):** 60 s timeout. 429, 5xx, timeouts and network errors are retried up to 4 times, waiting for `Retry-After` or the provider's "try again in" hint, otherwise exponential backoff (1, 2, 4, 8 s, capped at 30 s). A required wait over 60 s (for example an exhausted daily quota) fails immediately. 401/403, 404, token-limit and other 400 errors are not retried. Each failure type has its own exception class, and the API key is never included in an error message.
- **Latency logging:** every request appends one JSON line to `logs/rag_requests.jsonl` with embed, search, LLM, retry-wait, check and total times, plus status, scores, sources and token usage. It stores the question but not the answer text. A logging failure never breaks answering. `logs/` is git-ignored.
- **Rate limits:** response headers on this key reported 8,000 tokens and 1,000 requests. The evaluation pauses 15 s after each LLM request to stay under them.

### Limitations

- The grounding checks are lexical heuristics. They catch invented numbers, invalid citations and sentences unrelated to the sources, but cannot catch a wrong claim made of words the sources contain. They never fired in the 16-query run, so their detection rate on real model errors is untested and rests on unit tests alone. In this run the prompt's refusal rule is what stopped the fabricated-entity questions.
- The evaluation is 16 hand-written queries, one run, judged by keyword matching. It is a smoke test, not a measurement of hallucination rate, precision or recall.
- The relevance gate is leaky: off-topic or unanswerable queries scoring above 0.40 reach the model and depend on its refusal. The 0.40 threshold comes from Week 2's small benchmark. A higher cutoff would separate this particular sample (0.68 against 0.471), but it was not adopted because Week 2 showed natural-language queries scoring near 0.5.
- The corpus is a 7,000-article sample, so a valid question can be missing (for example "Who composed The Magic Flute?"). The system then refuses instead of answering.
- Latency figures come from one run on one machine and network, with 12 LLM requests. They depend on provider load and are not portable.
- Free-tier models and rate limits change. Check live model IDs with `scripts/list_llm_models.py` before running.
- Prompt injection was tested with a single case. It is not a security guarantee.

### Reproduce Week 3

```powershell
copy .env.example .env                  # then fill in your key, base URL and model ID
python scripts\list_llm_models.py       # shows the live model IDs for your key
python scripts\smoke_llm.py             # one test call
python scripts\ask.py "Who was the first person to walk on the moon?"
python scripts\run_rag_eval.py          # 16-query evaluation, about 3 minutes
python -m pytest -q                     # 191 tests, no API key needed
```

Requires the Chroma database from the Week 2 steps. The unit tests run without a key because the API is mocked.

## Project structure

```
data/raw/            raw downloads (git-ignored)
data/processed/      corpus, chunks, embeddings (git-ignored)
chroma_db/           persistent vector database (git-ignored)
docs/                chunking_strategy.md
reports/             embedding, ingest, retrieval benchmark, and RAG evaluation reports
scripts/             verify_env, download_data, inspect_raw, build_corpus,
                     build_chunks, embed_chunks, ingest_chroma, search_demo, list_llm_models, smoke_llm, ask, run_rag_eval,
                     benchmark_retrieval
src/                 preprocessing.py, chunking.py, vectorstore.py, llm_client.py, prompts.py, rag.py
tests/               pytest suite
```

## Data source

Articles come from the `wikimedia/wikipedia` dataset on Hugging Face (config `20231101.simple`). Wikipedia text is available under CC BY-SA and GFDL licenses; see the dataset card for details.

## Configuration

Copy `.env.example` to `.env` and fill in `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` (see Week 3). `.env` is git-ignored.
