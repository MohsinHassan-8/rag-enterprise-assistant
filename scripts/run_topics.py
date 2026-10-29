"""Topic modeling v2 (Week 4).

Date and year pages (for example "1323" or "October 23") are lists of events,
not topical articles, so they are set aside by a title rule (topic -2) and
BERTopic is fitted on the remaining documents. Saves
data/processed/doc_topics.parquet (git-ignored) and reports/topic_review.csv.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DATE_TOPIC = -2
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_TITLE = re.compile(
    r"^(\d{1,4}( BC| AD)?|\d{1,4}s( BC)?|(" + MONTHS + r") \d{1,2}"
    r"|\d{1,2}(st|nd|rd|th) (century|millennium)( BC)?)$"
)


def main():
    corpus = pd.read_parquet(PROC / "clean_corpus.parquet")
    index = pd.read_parquet(PROC / "doc_index.parquet")
    emb = np.load(PROC / "doc_embeddings.npy")

    assert len(index) == len(emb), "doc_index and doc_embeddings differ in length"
    assert index["id"].is_unique, "duplicate doc ids"
    corpus = corpus.set_index("id").loc[index["id"]].reset_index()
    assert (corpus["id"].to_numpy() == index["id"].to_numpy()).all()

    titles = corpus["title"].tolist()
    texts = corpus["text"].str.slice(0, 3000).tolist()
    is_date = np.array([bool(DATE_TITLE.match(t.strip())) for t in titles])
    fit_idx = np.where(~is_date)[0]
    print("documents:", len(texts), "| date/year pages set aside:", int(is_date.sum()))
    print("date page examples:", [titles[i] for i in np.where(is_date)[0][:12]])

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    model = BERTopic(embedding_model=None, umap_model=umap_model,
                     hdbscan_model=hdbscan_model, vectorizer_model=vectorizer,
                     top_n_words=10, calculate_probabilities=False, verbose=False)
    fitted, _ = model.fit_transform([texts[i] for i in fit_idx], embeddings=emb[fit_idx])

    topics = np.full(len(texts), -1, dtype=int)
    topics[fit_idx] = np.asarray(fitted)
    topics[is_date] = DATE_TOPIC

    pd.DataFrame({"id": index["id"], "title": titles, "topic": topics,
                  "is_date_page": is_date}).to_parquet(PROC / "doc_topics.parquet", index=False)

    n_out = int((topics == -1).sum())
    sizes = pd.Series(topics[topics >= 0]).value_counts()
    print("\ntopics found:", len(sizes))
    print("outliers (topic -1): %d (%.1f%% of all docs)" % (n_out, 100.0 * n_out / len(topics)))
    print("topic size: min %d median %d max %d" % (sizes.min(), int(sizes.median()), sizes.max()))
    rng = np.random.default_rng(42)
    out_idx = np.where(topics == -1)[0]
    print("outlier examples:", [titles[i] for i in rng.choice(out_idx, 15, replace=False)])

    rows = []
    print("\nAll topics, largest first")
    for t in sizes.index:
        members = np.where(topics == t)[0]
        centroid = emb[members].mean(axis=0)
        order = members[np.argsort(-(emb[members] @ centroid))[:3]]
        words = [w for w, _ in model.get_topic(int(t))]
        central = [titles[i] for i in order]
        rows.append({"topic": int(t), "size": int(sizes[t]), "label": "_".join(words[:3]),
                     "top_words": "; ".join(words), "central_titles": " | ".join(central)})
        print("[%d] n=%d | %s | %s" % (t, sizes[t], ", ".join(words[:6]), " / ".join(central[:2])))

    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS / "topic_review.csv", index=False, encoding="utf-8-sig")
    print("\nsaved doc_topics.parquet and reports/topic_review.csv")


if __name__ == "__main__":
    main()
