# RAG Enterprise Assistant

End-to-end Retrieval-Augmented Generation (RAG) pipeline: data ingestion, embedding, vector search, grounded LLM answers, NLP analysis, and a FastAPI service. Built week by week; this README is updated with each submission.

## Progress

| Week | Focus | Status |
|------|-------|--------|
| 1 | Environment, data ingestion, preprocessing | Done |
| 2 | Chunking, embeddings, vector database, retrieval benchmark | Done |
| 3 | LLM integration, prompt engineering, guarded RAG pipeline | Done |
| 4 | NLP analysis: topics, named entities, metadata filtering | Done |

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
python -m pytest -q                         # 221 tests (all weeks)
```

## Week 3: LLM Integration & Prompt Engineering

### Delivered

- `src/llm_client.py`: OpenAI-compatible chat client (httpx) with timeouts, retries with backoff, `Retry-After` handling, and typed errors (43 unit tests, HTTP mocked)
- `src/prompts.py`: system prompt, numbered-source context injection with a character budget, input cleaning, refusal detection (31 unit tests)
- `src/rag.py`: pipeline of retrieve, relevance gate, generate, verify, and log latency (40 unit tests, fake LLM)
- `scripts/ask.py` (command-line questions), `scripts/run_rag_eval.py` (16-query evaluation), `scripts/list_llm_models.py` and `scripts/smoke_llm.py` (check which model IDs are live and that the key works)
- `reports/rag_eval.json`: evaluation results. Per-request latency logs go to `logs/rag_requests.jsonl` (git-ignored)
- Test suite at the end of Week 3: 191 tests

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
python -m pytest -q                     # 221 tests, no API key needed
```

Requires the Chroma database from the Week 2 steps. The unit tests run without a key because the API is mocked.

## Week 4: NLP Analysis (Topics and Named Entities)

### Delivered

- `scripts/build_doc_embeddings.py`: one normalized vector per document (the mean of its chunk vectors), after checking that the stored embeddings match the chunk order
- `scripts/run_topics.py`: BERTopic topic modeling on those vectors, with date and year pages set aside by a title rule. Writes `reports/topic_review.csv`
- `src/topic_themes.py` and `scripts/build_doc_themes.py`: hand-made mapping of the 109 topics to 25 broad themes (`reports/theme_summary.csv`)
- Manual validation: `scripts/sample_for_validation.py` and `scripts/score_theme_validation.py` (topic themes), `scripts/sample_ner_validation.py` and `scripts/score_ner_validation.py` (NER). Samples and labels are in `reports/`
- `scripts/extract_places.py` and `src/place_utils.py`: place entities (GPE, LOC) per document with spaCy, normalised
- `scripts/enrich_chroma.py`: writes theme, topic id and places into the metadata of all 34,509 chunks in ChromaDB, then verifies it (`reports/metadata_log.json`)
- `src/vectorstore.py`: `build_where()` for filters and `update_metadata()`. `src/rag.py` and `scripts/ask.py`: optional `theme` and `place` filters
- `scripts/benchmark_filters.py`: filtered-search latency, and a check of whether filters hide or surface the expected article (`reports/filter_benchmark.json`)
- Test suite: 221 tests in total across all weeks (30 new: 6 topic themes, 5 place utilities, 14 vector store filters, 5 filtered RAG)

### Measured results

All figures come from single runs on one machine (see `reports/`). Topic and NER accuracy were measured on small hand-checked samples, so every figure has a wide margin.

**Topic modeling** (BERTopic 0.17.4 on per-document vectors)

| Measure | Result |
|---------|--------|
| Documents | 6,960 (686 date and year pages set aside, 6,274 modeled) |
| Topics found | 109 (size min 15, median 29, max 415) |
| Outliers (topic -1) | 1,276 (18.3% of all documents) |
| Documents with a broad theme | 4,998 (71.8%), in 25 themes |
| Documents without a theme | 1,962 (28.2%): 1,276 outliers plus 686 date pages |
| Chunks with a theme | 22,102 of 34,509 (64.0%). Date pages account for 7,392 chunks (21.4%) and outliers for 5,015 (14.5%) |

- The first run, without the date rule, also gave 109 topics and 18.2% outliers, but titles such as "1323" and "October 23" were central members of the war, kings and actors topics. A title rule now sets those pages aside as "Dates and years".
- Because date pages are long event lists, a theme filter searches only about two thirds of the index.

**Manual validation of the themes**

| Sample | Result |
|--------|--------|
| 50 random themed documents, lenient (borderline counted correct) | 48/50 = 96.0% (95% Wilson interval 86.5% to 98.9%) |
| Same, strict (borderline counted wrong) | 41/50 = 82.0% (69.2% to 90.2%) |
| Wrong theme (2) | Theatre of ancient Greece (put in Film, TV and anime), Christmas (put in Math, units and time) |
| 10 random unclassified outliers | 5 clearly belonged to one of the 25 themes (for example September 11 attacks, Cervantes), 2 borderline, 3 genuine outliers |

- Verdicts were proposed with the help of an AI assistant from each article's title and first 150 characters, and were not independently re-labelled. Treat them as an estimate, not an independent measurement. The labels are in `reports/theme_validation_labels.csv` and can be edited in `scripts/score_theme_validation.py`.

**Named entity recognition** (spaCy `en_core_web_sm`, strict scoring: span and type must both be right)

40 sentences, one per article, drawn with a fixed seed from non-date articles, containing 53 gold entities and 58 predictions.

| Type | TP | FP | FN | Precision | Recall | F1 |
|------|----|----|----|-----------|--------|----|
| PERSON | 9 | 7 | 4 | 56.2% | 69.2% | 62.1% |
| ORG | 6 | 12 | 3 | 33.3% | 66.7% | 44.4% |
| GPE | 18 | 3 | 10 | 85.7% | 64.3% | 73.5% |
| LOC | 2 | 1 | 1 | 66.7% | 66.7% | 66.7% |
| All | 35 | 23 | 18 | 60.3% | 66.0% | 63.1% |

- LOC rests on 3 gold entities, so it is not a measurement. The entity verdicts were proposed with an AI assistant from the sentence alone and were not independently checked.
- ORG is often applied to places and products (Cumbria, Lancashire, Hackintosh), and PERSON to place names and software (Makkah, macOS). Only GPE was reliable enough to filter on, so **only GPE and LOC are stored as place metadata**.
- Some documents still contain Wikipedia category lines (for example "1910 establishments in the United Kingdom"), which the Week 1 cleaning did not remove. They appear in at least 3 of the 40 sampled sentences and produce odd entity spans.
- Place extraction covers the first 2,000 characters of each document. 5,383 of 6,960 documents (77.3%) have at least one place. After normalisation (`U.S.` and `USA` become `United States`, `UK` becomes `United Kingdom`, and `Earth`, `Middle Ages` and `Sun` are dropped), 28,782 of 34,509 chunks (83.4%) carry at least one place.

**Metadata and filtered retrieval**

Every chunk now stores `theme`, `topic_id`, `places` and five flat slots `place_1` to `place_5`. A place filter is an OR across the five slots. All 34,509 chunks were updated in place in 14.5 s, and the per-theme counts in the database were checked against the expected counts.

Search latency, k=5, 10 repetitions per query, 12 queries (place rows use the 4 queries that have a place):

| Search | n | p50 (ms) | p95 (ms) | Chunks matched by the filter |
|--------|---|----------|----------|------------------------------|
| Unfiltered | 120 | 7.44 | 14.32 | all 34,509 |
| Theme | 120 | 20.03 | 28.37 | 447 to 1,957 |
| Place | 40 | 33.43 | 40.45 | 134 to 4,268 |
| Theme + place | 40 | 22.92 | 32.74 | 42 to 357 |

- Filtered search was about 2.7x to 4.5x slower at p50, an extra 13 to 26 ms. This matches the Week 2 observation (27.8 ms filtered against 4.6 ms unfiltered). The cause was not investigated. The sample is too small to tell whether cost follows the number of matching chunks.
- Effect on results, 12 questions with the correct theme (and place where it applies) chosen by hand: the expected article was in the top 5 both with and without the filter in 9 cases, only with the filter in 1 (Oliver Twist), and the filter never hid it. In 2 cases the expected article is not in the 7,000-article sample ("Neutron star", "Rhine"), so they are corpus gaps.
- The top score was identical in 11 of the 12 cases, so the filter acts as a restriction ("search only this theme") and did not change the ranking in this test. This benchmark does not show filters improving precision.
- Live checks with `scripts/ask.py`: "Who was Henry VIII?" with `--theme History --place England` returned status `ok` with 5 History sources that all mention England. A solar eclipse question restricted to Music scored 0.239 and was stopped by the relevance gate (no LLM call). An unknown theme returned a clear "no passages match" message without calling the LLM.

### Approach and design decisions

- **Topics per document, not per chunk:** topic modeling uses the mean of each document's chunk vectors, so nothing is re-embedded and the 6,960 documents are clustered instead of 34,509 chunks. The stored chunk embeddings were first checked against fresh embeddings of 30 random chunks (cosine 1.0000).
- **Clustering setup:** UMAP (5 components, 15 neighbors, cosine, seed 42) then HDBSCAN (minimum cluster size 15, minimum samples 5). Topic keywords come from the first 3,000 characters of each document with English stop words and 1 to 2 word phrases.
- **Date and year pages:** documents whose title is a year, a decade, a century or a month and day (for example "1323", "October 23") are set aside by a title rule, because they are event lists and polluted the topics.
- **Broad themes:** the 109 topics contain near-duplicates (several music, war and football topics), so each topic id was mapped by hand to one of 25 themes, which are the values used for filtering. Unit tests check that every id from 0 to 108 is mapped exactly once.
- **NER instead of sentiment:** Simple English Wikipedia is expected to be mostly neutral, so sentiment would make a weak evaluation (not measured). NER gives place filters, and the validation showed which entity types are usable.
- **Place cleaning:** a leading "the" is removed, spans containing a newline or longer than 40 characters are dropped, the 5 most frequent places per document are kept, and aliases and a small stoplist are applied.
- **Flat metadata:** values are stored as strings and ints with five place slots, and `build_where()` combines theme, topic id and place with AND, and the five place slots with OR.
- **Safe updates:** `update_metadata()` changes metadata only, so vectors and text are untouched. The verification read pages through the collection 2,000 chunks at a time, because one read of all chunks exceeds a SQLite variable limit inside Chroma.
- **Filter behavior in the pipeline:** an invalid filter returns `invalid_query` and a filter that matches nothing returns a clear message, both without calling the LLM. The relevance gate still applies to filtered searches.

### Limitations

- The theme and NER verdicts were proposed with an AI assistant and not independently re-labelled. The samples are small (50 themed documents, 10 outliers, 40 sentences with 53 gold entities), so intervals are wide.
- 28.2% of documents (36.0% of chunks, because date pages are long) have no theme. Half of the sampled outliers clearly belonged to a theme, so "Unclassified" understates what could be labelled. Reassigning outliers to the nearest theme was not tried.
- Themes are a manual mapping from topic ids and some are mixed: Film, TV and anime absorbed theatre, fiction and broadcasting, Math, units and time contains month pages, and one Europe topic is mostly Polish and Hungarian school stubs. The mapping belongs to the saved run, so re-running `run_topics.py` can change topic ids and break it.
- Short documents and jargon were handled only for date and year pages. Disambiguation pages (for example "Halifax", "Queen Elizabeth") were not handled separately, and short documents were not analysed on their own.
- Metadata is per document and copied to every chunk. Places come from the first 2,000 characters, so a chunk deep in a long article can carry places it does not mention and can miss places it does.
- GPE recall was 64.3% in the NER sample, so a place filter misses documents. "America", "Britain" and "England" are not merged with "United States" or "United Kingdom". PERSON and ORG are not stored because their precision was too low.
- Category lines from Wikipedia survive the Week 1 cleaning and add noise to NER and topic keywords.
- Filter benchmark: 12 cases with correct filters chosen by hand and no test with a wrong filter. A filtered query can land close to the relevance gate (Oliver Twist scored 0.416 with the filter against a 0.40 threshold). Filtered search is slower and the cause is unknown.
- Sentiment analysis was not implemented. The `topic_id` filter exists in `build_where()` but is not exposed in the pipeline or `ask.py`.
- Rebuilding ChromaDB with `ingest_chroma.py --reset` removes the metadata, so run `enrich_chroma.py` again afterwards.

### Reproduce Week 4

```powershell
python scripts\build_doc_embeddings.py   # -> data/processed/doc_embeddings.npy, doc_index.parquet
python scripts\run_topics.py             # -> doc_topics.parquet, reports/topic_review.csv
python scripts\build_doc_themes.py       # -> doc_themes.parquet, reports/theme_summary.csv
python scripts\score_theme_validation.py # accuracy from reports/theme_validation_sample.csv
python scripts\score_ner_validation.py   # precision and recall from reports/ner_validation_sample.csv
python scripts\extract_places.py         # -> data/processed/doc_places.parquet
python scripts\enrich_chroma.py          # writes the metadata into ChromaDB
python scripts\ask.py "Who was Henry VIII?" --theme History --place England
python scripts\benchmark_filters.py      # -> reports/filter_benchmark.json
python -m pytest -q                      # 221 tests
```

Topic ids can differ if `run_topics.py` is re-run on another machine or library version, and `src/topic_themes.py` maps the ids of the saved run. Requires the Chroma database from the Week 2 steps and, for `ask.py`, the API key from Week 3.

## Project structure

```
data/raw/            raw downloads (git-ignored)
data/processed/      corpus, chunks, embeddings (git-ignored)
chroma_db/           persistent vector database (git-ignored)
docs/                chunking_strategy.md
reports/             embedding, ingest, retrieval benchmark, RAG evaluation, topic, NER and filter reports
scripts/             verify_env, download_data, inspect_raw, build_corpus,
                     build_chunks, embed_chunks, ingest_chroma, search_demo, list_llm_models, smoke_llm, ask, run_rag_eval,
                     build_doc_embeddings, run_topics, build_doc_themes, sample_for_validation,
                     score_theme_validation, sample_ner_validation, score_ner_validation,
                     extract_places, enrich_chroma, benchmark_filters,
                     benchmark_retrieval
src/                 preprocessing.py, chunking.py, vectorstore.py, llm_client.py, prompts.py, rag.py, topic_themes.py, place_utils.py
tests/               pytest suite
```

## Data source

Articles come from the `wikimedia/wikipedia` dataset on Hugging Face (config `20231101.simple`). Wikipedia text is available under CC BY-SA and GFDL licenses; see the dataset card for details.

## Configuration

Copy `.env.example` to `.env` and fill in `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` (see Week 3). `.env` is git-ignored.
