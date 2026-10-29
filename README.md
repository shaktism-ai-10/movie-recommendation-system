# Movie Recommendation System

A simple content-based movie recommender built for our **Machine Learning** course at **Sri Sri University**.
Give it one movie you liked and it suggests the **5 most similar movies**.

**Guided by:** Ansh Agrawal Sir

| Team member | Roll No. |
|---|---|
| Shakti Subham Mishra | 1ETBTA20251040 |
| Apurba Mohanty | 1ETBTA20251039 |
| Preksha Singh | 1ETBTA20251038 |
| Puravi Dash | 1ETBTA20251041 |

## How it works

1. **Load data:** `movies.csv` holds 110 movies (1968 to 2022), each with genres, director, 2 lead actors and 4 to 6 plot keywords.
2. **Build tags:** these fields are joined into one line of words per movie. Names become one word (`christophernolan`) so different people are never confused.
3. **Vectorise:** `CountVectorizer` turns each line into word counts (bag of words, 537-word vocabulary).
4. **Similarity:** `cosine_similarity` compares every movie with every other movie (110 x 110 table).
5. **Recommend:** sort the chosen movie's row and return the top 5 with a match percentage.

Example: *The Dark Knight* (11 words) and *Batman Begins* (10 words) share 7 words, so similarity = 7 / sqrt(11 x 10) = **0.67**, shown as a 67% match.

## How to run

```bash
pip install -r requirements.txt
python movie_recommender.py "The Dark Knight"
python movie_recommender.py --evaluate
```

Typos are handled: `python movie_recommender.py "toy stroy"` still finds Toy Story.

Sample output:

```
Because you liked 'The Dark Knight', you may also like:
  1. Batman Begins              match 67%
  2. The Dark Knight Rises      match 60%
  3. Joker                      match 36%
  4. The Prestige               match 29%
  5. The Godfather              match 20%
```

## Results

Tested on 20 movies where we wrote down in advance which movies a fan would expect (26 expected movies in total):

| Model | Precision | Recall | F1-score | Hit Rate@5 |
|---|---|---|---|---|
| Random picks (mean of 100 runs) | 0.013 | 0.048 | 0.020 | 0.062 |
| Genre-only model | 0.200 | 0.769 | 0.317 | 0.800 |
| **Our model (all tags)** | **0.260** | **1.000** | **0.413** | **1.000** |

Our model found all 26 expected movies. Precision 0.26 is the highest possible here, since only 26 expected answers exist for 100 recommendation slots.

## Files

| File | What it is |
|---|---|
| `movie_recommender.py` | The model, command-line demo and evaluation |
| `movies.csv` | Dataset of 110 movies prepared by the team |
| `requirements.txt` | Python libraries needed |
| `docs/Movie_Recommendation_Report.pdf` | Full project report |
| `docs/Movie_Recommendation_Work_Report.pdf` | Summary of the work done |
| `docs/Movie_Recommendation_Presentation.pdf` | 13-slide presentation |
| `docs/Movie_Recommendation_SIH_Format.pdf` | 6-slide idea-format presentation |

## Limitations and future work

- Quality depends on the hand-written tags; the dataset (110 movies) and test set (20 movies) are small.
- It recommends "more of the same" and does not learn from user ratings.
- Next steps: a larger dataset (MovieLens, TMDB 5000), TF-IDF or embeddings on plot summaries, a hybrid with user ratings, and a small web app.
