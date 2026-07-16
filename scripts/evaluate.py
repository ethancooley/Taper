"""Evaluation for Taper: naive vs responsible recommendations.

Produces:
  data/outputs/eval_comparison.png  - 6-week load trajectory + diversity chart
  Printed precision@5 for both systems (relevance cost of the constraint)

Simulation: an "eager runner" follows the recommender's top pick every day
for 6 weeks. The naive system amplifies the athlete's intensity bias and
sends acute:chronic workload ratio (ACWR) into the injury-risk zone; the
constrained system tracks a safe ramp.

Attribution: This file was written with the assistance of Claude (Anthropic).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import (AthleteState, ResponsibleReranker, WorkoutRecommender,
                   naive_top_k, ACWR_HIGH, ACWR_DANGER)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def precision_at_k(recommender: WorkoutRecommender,
                   interactions: pd.DataFrame, k: int = 5,
                   reranker: ResponsibleReranker | None = None) -> float:
    """Leave-last-week-out precision@k over all athletes."""
    hits, total = 0, 0
    last_week = interactions.week.max()
    for athlete_id, group in interactions.groupby("athlete_id"):
        train = group[group.week < last_week]
        test = set(group[group.week == last_week].workout_id)
        if train.empty or not test:
            continue
        scored = recommender.score(history_ids=list(train.workout_id))
        if reranker:
            state = _state_from_history(train, recommender.catalog)
            top = reranker.rerank(scored, state, top_k=k)
        else:
            top = naive_top_k(scored, top_k=k)
        hits += len(set(top.workout_id) & test)
        total += k
    return hits / total


def _state_from_history(history: pd.DataFrame,
                        catalog: pd.DataFrame) -> AthleteState:
    state = AthleteState()
    merged = history.merge(catalog, on="workout_id")
    for _, row in merged.sort_values(["week", "day"]).iterrows():
        state.record(row.load, row.workout_type)
    return state


def simulate_six_weeks(recommender: WorkoutRecommender, catalog: pd.DataFrame,
                       use_reranker: bool) -> tuple[list[float], list[str]]:
    """Athlete follows the top recommendation ~5 days/week for 6 weeks."""
    reranker = ResponsibleReranker()
    state = AthleteState()
    # Low training base: two weeks of short, sparse easy running.
    easy = catalog[(catalog.sport == "run") & (catalog.intensity <= 2)]
    for i in range(14):
        if i % 3 == 0:
            w = easy.sample(1, random_state=i).iloc[0]
            state.record(w.load * 0.6, w.workout_type)
        else:
            state.record(0.0, "rest")
    # But the athlete's *stated taste* (interaction history) skews hard:
    # they favorite interval and tempo sessions, like the eager_runner archetype.
    hard = catalog[(catalog.sport == "run") & (catalog.intensity >= 4)]
    history = list(hard.sample(4, random_state=1).workout_id) + \
        list(easy.sample(1, random_state=1).workout_id)

    acwr_series, types = [], []
    for day in range(42):
        if day % 7 in (5, 6):  # two rest days a week
            state.record(0.0, "rest")
            acwr_series.append(state.acute_load() / max(state.chronic_load(), 1e-9))
            continue
        scored = recommender.score(history_ids=history)
        if use_reranker:
            pick = reranker.rerank(scored, state, top_k=1).iloc[0]
        else:
            pick = naive_top_k(scored, top_k=1).iloc[0]
        state.record(pick.load, pick.workout_type)
        history.append(int(pick.workout_id))
        types.append(pick.workout_type)
        acwr_series.append(state.acute_load() / max(state.chronic_load(), 1e-9))
    return acwr_series, types


def make_figure(acwr_naive, acwr_safe, types_naive, types_safe,
                p_naive, p_safe, out_path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    days = np.arange(len(acwr_naive))
    ax1.axhspan(ACWR_HIGH, ACWR_DANGER, color="orange", alpha=0.15,
                label="elevated risk")
    ax1.axhspan(ACWR_DANGER, 3.0, color="red", alpha=0.15,
                label="injury-risk zone")
    ax1.plot(days, acwr_naive, color="crimson", lw=2,
             label=f"naive (P@5={p_naive:.2f})")
    ax1.plot(days, acwr_safe, color="seagreen", lw=2,
             label=f"constrained (P@5={p_safe:.2f})")
    ax1.set_ylim(0.5, 3.0)
    ax1.set_xlabel("day")
    ax1.set_ylabel("acute:chronic workload ratio")
    ax1.set_title("Training load trajectory over 6 weeks")
    ax1.legend(loc="upper left", fontsize=9)

    all_types = sorted(set(types_naive) | set(types_safe))
    naive_counts = [types_naive.count(t) for t in all_types]
    safe_counts = [types_safe.count(t) for t in all_types]
    x = np.arange(len(all_types))
    ax2.bar(x - 0.2, naive_counts, 0.4, color="crimson", label="naive")
    ax2.bar(x + 0.2, safe_counts, 0.4, color="seagreen", label="constrained")
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_types, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("times recommended")
    ax2.set_title("Workout variety")
    ax2.legend()

    fig.suptitle("Taper: relevance vs responsible recommendations", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


def main() -> None:
    catalog = pd.read_csv(os.path.join(DATA_DIR, "raw", "workouts.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "raw", "interactions.csv"))

    rec = WorkoutRecommender(catalog).fit(interactions)

    p_naive = precision_at_k(rec, interactions)
    p_safe = precision_at_k(rec, interactions, reranker=ResponsibleReranker())
    print(f"precision@5  naive={p_naive:.3f}  constrained={p_safe:.3f}")

    acwr_naive, types_naive = simulate_six_weeks(rec, catalog, use_reranker=False)
    acwr_safe, types_safe = simulate_six_weeks(rec, catalog, use_reranker=True)
    n_viol_naive = sum(a > ACWR_DANGER for a in acwr_naive)
    n_viol_safe = sum(a > ACWR_DANGER for a in acwr_safe)
    print(f"days above ACWR {ACWR_DANGER}: naive={n_viol_naive}  constrained={n_viol_safe}")

    out_dir = os.path.join(DATA_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    make_figure(acwr_naive, acwr_safe, types_naive, types_safe,
                p_naive, p_safe, os.path.join(out_dir, "eval_comparison.png"))


if __name__ == "__main__":
    main()
