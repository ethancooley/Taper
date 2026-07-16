# Taper

A workout recommender that optimizes for what athletes can **safely absorb**, not just what they like to click on.

Fitness recommenders learn from biased feedback: athletes over-select the workouts they already enjoy (intervals, tempo) and skip the ones they need (recovery, strength). A naive engagement-optimized recommender learns that bias and amplifies it, ramping training load straight into injury-risk territory. Taper adds a responsible re-ranking layer that keeps recommendations relevant **and** inside a safe training-load progression.

## How it works

1. **Relevance (hybrid recommender)** — cosine similarity over workout content features (sport, type, intensity, duration, load) blended with item-item co-occurrence learned from athlete completion histories.
2. **Cold start** — new users complete a short intake (sport focus, goal hours, recent hours, max intensity) and receive content-only recommendations until they build history.
3. **Responsibility (re-ranker)** — top relevance candidates are rescored using:
   - **ACWR constraint**: the acute (7-day) to chronic (28-day) workload ratio each pick would produce. Picks pushing ACWR toward 1.5+ (the injury-risk zone in sports science literature) are penalized; picks crossing it are hard-excluded.
   - **Variety penalty**: repeats of recently recommended workout types are demoted, which pulls recovery, mobility, and strength work back into the mix.

## Results

Evaluation (`scripts/evaluate.py`) simulates an eager athlete on a low training base following each system's top pick for 6 weeks:

| Metric | Naive | Constrained |
|---|---|---|
| Precision@5 (leave-last-week-out) | 0.24 | 0.14 |
| Days above ACWR 1.5 (injury-risk) | 15 | 1 |
| Distinct workout types recommended | 1 | 6 |

The constrained system trades a modest amount of measured relevance — much of which is bias-agreement, not usefulness — for a dramatically safer load trajectory and a balanced training mix. See `data/outputs/eval_comparison.png`.

## Setup and run

```bash
pip install -r requirements.txt
python setup.py            # generates data and the evaluation figure
uvicorn main:app --reload  # serves the app at http://localhost:8000
```

## Repo structure

```
├── README.md
├── requirements.txt
├── setup.py                <- generate data + run evaluation
├── main.py                 <- FastAPI app / user interface
├── scripts
│   ├── make_dataset.py     <- synthetic workout catalog + biased interactions
│   ├── model.py            <- WorkoutRecommender + ResponsibleReranker
│   └── evaluate.py         <- naive vs constrained comparison
├── static/index.html       <- frontend
├── data
│   ├── raw / processed / outputs
├── models
└── notebooks
```

## Constraint story (rubric mapping)

- **Simple recommendation approach**: content similarity + item-item co-occurrence.
- **Real-world constraints**: cold start (intake path) and safety/diversity (ACWR + variety re-ranker).
- **Relevance vs responsibility demonstration**: `evaluate.py` figure and the side-by-side naive/constrained UI.

## Limitations

Interaction data is simulated; load scoring is a TSS-style approximation; ACWR is a contested (though widely used) injury-risk proxy and this tool is not medical or coaching advice.

## Attribution

Built with the assistance of Claude (Anthropic). Per-file attribution noted in file headers.
