"""
Movie Recommendation System - Content-Based Filtering
(Bag-of-Words + Cosine Similarity)

Idea: describe every movie as a bag of words ("tags") made from its
genres, director, main cast and plot keywords. Two movies whose tags
overlap a lot are "similar". To recommend, we pick the movies with the
highest cosine similarity to the one the user liked.

Usage:
    python movie_recommender.py "The Dark Knight"    -> top-5 similar movies
    python movie_recommender.py --evaluate           -> metrics vs baselines
    python movie_recommender.py                      -> asks for a title
"""
import os
import sys
import random
import difflib

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = 5
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies.csv")


def one_token(name):
    """'Christopher Nolan' -> 'christophernolan' so a person is one word, not two."""
    return name.replace(" ", "").replace(".", "").replace("-", "").lower()


def load_movies(path=DATA):
    df = pd.read_csv(path)
    genres = df["genres"].str.replace("|", " ")
    director = df["director"].map(one_token)
    cast = df["cast"].str.split("|").map(lambda names: " ".join(one_token(n) for n in names))
    df["tags"] = genres + " " + director + " " + cast + " " + df["keywords"]
    df["genre_tags"] = genres  # used only by the genre-only baseline
    return df


def build_similarity(df, column="tags"):
    """Step 1: text -> word-count vectors.  Step 2: cosine similarity between every pair."""
    vectors = CountVectorizer().fit_transform(df[column])
    return cosine_similarity(vectors)


def find_title(query, df):
    """Exact (case-insensitive) match first, then closest spelling, so typos still work."""
    titles = {t.lower(): t for t in df["title"]}
    if query.lower() in titles:
        return titles[query.lower()]
    close = difflib.get_close_matches(query.lower(), titles.keys(), n=1, cutoff=0.5)
    if not close:
        raise ValueError(f"Movie '{query}' not found in the dataset.")
    return titles[close[0]]


def recommend(query, df, sim, k=TOP_K):
    title = find_title(query, df)
    i = df.index[df["title"] == title][0]
    ranked = sorted(enumerate(sim[i]), key=lambda pair: -pair[1])  # highest similarity first
    return title, [(df["title"][j], float(score)) for j, score in ranked if j != i][:k]


# ---------------------------------------------------------------------------
# Evaluation: for 20 test movies we wrote down which movies a fan would clearly
# expect to see (sequels, same universe, same director/style). A model "hits"
# when those movies appear in its top-5.
# ---------------------------------------------------------------------------
TEST_SET = {
    "The Dark Knight": ["Batman Begins", "The Dark Knight Rises"],
    "Terminator 2: Judgment Day": ["The Terminator"],
    "The Matrix": ["The Matrix Reloaded"],
    "Avengers: Endgame": ["Avengers: Infinity War", "The Avengers"],
    "Iron Man": ["The Avengers"],
    "The Godfather": ["The Godfather Part II"],
    "Goodfellas": ["The Godfather", "The Departed"],
    "The Lord of the Rings: The Fellowship of the Ring":
        ["The Lord of the Rings: The Two Towers", "The Lord of the Rings: The Return of the King"],
    "The Hobbit: An Unexpected Journey": ["The Lord of the Rings: The Fellowship of the Ring"],
    "Harry Potter and the Sorcerer's Stone":
        ["Harry Potter and the Prisoner of Azkaban", "Harry Potter and the Deathly Hallows: Part 2"],
    "Toy Story": ["Toy Story 3"],
    "Star Wars: A New Hope": ["The Empire Strikes Back"],
    "Alien": ["Aliens"],
    "Spirited Away": ["My Neighbor Totoro"],
    "Casino Royale": ["Skyfall"],
    "Se7en": ["Zodiac"],
    "The Silence of the Lambs": ["Se7en", "Zodiac"],
    "3 Idiots": ["PK"],
    "Baahubali: The Beginning": ["RRR"],
    "La La Land": ["Whiplash"],
}


def confusion(df, get_top_k):
    """Treat each test query as a yes/no task over all other movies:
    predicted = in top-5, actual = in the expected list."""
    tp = fp = fn = tn = hits = 0
    for query, expected in TEST_SET.items():
        predicted = set(get_top_k(query))
        expected = set(expected)
        others = len(df) - 1
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        tn += others - len(predicted | expected)
        hits += bool(predicted & expected)
    return tp, fp, fn, tn, hits


def metrics(tp, fp, fn, tn, hits):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        "Accuracy": (tp + tn) / (tp + fp + fn + tn),
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "HitRate@5": hits / len(TEST_SET),
    }


def evaluate(path=DATA):
    # Shuffle so that ties are not broken by CSV order (sequels sit next to each other there).
    df = load_movies(path).sample(frac=1, random_state=42).reset_index(drop=True)
    full_sim = build_similarity(df, "tags")
    genre_sim = build_similarity(df, "genre_tags")
    titles = list(df["title"])

    def random_top_k(query, rng):
        return rng.sample([t for t in titles if t != query], TOP_K)

    # Random baseline: average of 100 runs so one lucky draw doesn't mislead.
    runs = [confusion(df, lambda q, rng=random.Random(seed): random_top_k(q, rng)) for seed in range(100)]
    random_cm = tuple(sum(r[i] for r in runs) / len(runs) for i in range(5))

    results = {
        "Random baseline": random_cm,
        "Genre-only model": confusion(df, lambda q: [t for t, _ in recommend(q, df, genre_sim)[1]]),
        "Proposed (full tags)": confusion(df, lambda q: [t for t, _ in recommend(q, df, full_sim)[1]]),
    }
    print(f"{'Model':<22}{'TP':>6}{'FP':>7}{'FN':>6}{'TN':>8}  " + "  ".join(f"{m:>9}" for m in metrics(1, 1, 1, 1, 1)))
    for name, cm in results.items():
        m = metrics(*cm)
        print(f"{name:<22}{cm[0]:>6.1f}{cm[1]:>7.1f}{cm[2]:>6.1f}{cm[3]:>8.1f}  " + "  ".join(f"{v:>9.3f}" for v in m.values()))

    # Self-check: the proposed model must beat both baselines on recall.
    assert metrics(*results["Proposed (full tags)"])["Recall"] > metrics(*results["Genre-only model"])["Recall"]
    assert metrics(*results["Genre-only model"])["Recall"] > metrics(*results["Random baseline"])["Recall"]
    return results


def main():
    if sys.argv[1:] == ["--evaluate"]:
        evaluate()
        return
    query = " ".join(sys.argv[1:]) or input("Enter a movie you liked: ")
    df = load_movies()
    sim = build_similarity(df)
    try:
        title, recs = recommend(query, df, sim)
    except ValueError as err:
        print(err)
        return
    print(f"\nBecause you liked '{title}', you may also like:")
    for rank, (movie, score) in enumerate(recs, 1):
        print(f"  {rank}. {movie:<55} match {score:.0%}")


if __name__ == "__main__":
    main()
