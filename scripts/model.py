"""Recommendation models for Taper.

WorkoutRecommender: hybrid relevance scoring (content similarity + item-item
co-occurrence from interaction history), with a content-only cold-start path.

ResponsibleReranker: re-ranks relevance candidates using an acute:chronic
workload ratio (ACWR) constraint and a variety penalty, so recommendations
stay inside a safe training-load ramp instead of amplifying athlete bias.

Attribution: This file was written with the assistance of Claude (Anthropic).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ACWR_LOW = 0.8
ACWR_HIGH = 1.3
ACWR_DANGER = 1.5


class WorkoutRecommender:
    """Hybrid content + co-occurrence recommender over a workout catalog."""

    def __init__(self, catalog: pd.DataFrame, content_weight: float = 0.5):
        self.catalog = catalog.reset_index(drop=True)
        self.content_weight = content_weight
        self._content_sim = self._build_content_similarity()
        self._cooc_sim = None

    def _build_content_similarity(self) -> np.ndarray:
        cat = OneHotEncoder(sparse_output=False).fit_transform(
            self.catalog[["sport", "workout_type"]])
        num = StandardScaler().fit_transform(
            self.catalog[["intensity", "duration_min", "load"]])
        features = np.hstack([cat, num * 0.5])
        return cosine_similarity(features)

    def fit(self, interactions: pd.DataFrame) -> "WorkoutRecommender":
        """Learn item-item co-occurrence similarity from completion history."""
        n = len(self.catalog)
        user_item = np.zeros((interactions.athlete_id.nunique(), n))
        for (a, w), cnt in interactions.groupby(["athlete_id", "workout_id"]).size().items():
            user_item[a, w] = cnt
        cooc = cosine_similarity(user_item.T)
        np.fill_diagonal(cooc, 0)
        self._cooc_sim = cooc
        return self

    def _profile_vector(self, history_ids: list[int]) -> np.ndarray:
        """Blend content and co-occurrence similarity to the user's history."""
        content = self._content_sim[history_ids].mean(axis=0)
        if self._cooc_sim is None:
            return content
        cooc = self._cooc_sim[history_ids].mean(axis=0)
        return self.content_weight * content + (1 - self.content_weight) * cooc

    def score(self, history_ids: list[int] | None = None,
              intake: dict | None = None) -> pd.DataFrame:
        """Return the catalog with a relevance score for this athlete.

        history_ids: workout_ids the athlete has completed (warm start).
        intake: cold-start dict, e.g. {"sports": ["run"], "max_intensity": 4}.
        """
        if history_ids:
            scores = self._profile_vector(history_ids)
        elif intake:
            scores = self._score_from_intake(intake)
        else:
            raise ValueError("Need interaction history or an intake form.")
        out = self.catalog.copy()
        out["relevance"] = scores
        return out.sort_values("relevance", ascending=False)

    def _score_from_intake(self, intake: dict) -> np.ndarray:
        """Cold start: content-only scoring from stated preferences."""
        sports = intake.get("sports", ["run"])
        max_intensity = intake.get("max_intensity", 5)
        scores = np.where(self.catalog.sport.isin(sports), 1.0, 0.2)
        scores = np.where(self.catalog.intensity > max_intensity, 0.0, scores)
        # Mild preference for mid-intensity when we know nothing else.
        scores = scores * (1 - 0.05 * abs(self.catalog.intensity - 3))
        return scores


@dataclass
class AthleteState:
    """Rolling training-load state used by the re-ranker."""
    daily_loads: list[float] = field(default_factory=list)  # last 28 days
    recent_types: list[str] = field(default_factory=list)   # last 7 workouts

    def acute_load(self) -> float:
        return float(np.mean(self.daily_loads[-7:])) if self.daily_loads else 0.0

    def chronic_load(self) -> float:
        return float(np.mean(self.daily_loads[-28:])) if self.daily_loads else 0.0

    def acwr_if_added(self, load: float) -> float:
        acute = (sum(self.daily_loads[-6:]) + load) / 7.0
        chronic = (sum(self.daily_loads[-27:]) + load) / 28.0
        return acute / chronic if chronic > 0 else 1.0

    def record(self, load: float, workout_type: str) -> None:
        self.daily_loads.append(load)
        self.daily_loads = self.daily_loads[-28:]
        self.recent_types.append(workout_type)
        self.recent_types = self.recent_types[-7:]


class ResponsibleReranker:
    """Re-rank relevance candidates under injury-risk and variety constraints."""

    def __init__(self, acwr_penalty: float = 2.0, variety_penalty: float = 0.3):
        self.acwr_penalty = acwr_penalty
        self.variety_penalty = variety_penalty

    def rerank(self, candidates: pd.DataFrame, state: AthleteState,
               top_k: int = 5) -> pd.DataFrame:
        """Score = relevance - ACWR risk penalty - repetition penalty."""
        cands = candidates.copy()
        # Normalize relevance to [0, 1] so penalties are comparable.
        rel = cands.relevance
        cands["rel_norm"] = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)

        cands["acwr_after"] = cands.load.apply(state.acwr_if_added)
        risk = np.clip((cands.acwr_after - ACWR_HIGH) / (ACWR_DANGER - ACWR_HIGH), 0, None)
        repeat_counts = cands.workout_type.map(
            lambda t: state.recent_types.count(t))

        cands["final_score"] = (cands.rel_norm
                                - self.acwr_penalty * risk
                                - self.variety_penalty * repeat_counts)
        # Hard-exclude anything that would cross the danger threshold.
        safe = cands[cands.acwr_after <= ACWR_DANGER]
        if len(safe) >= top_k:
            cands = safe
        cands = cands.sort_values("final_score", ascending=False)
        return cands.drop_duplicates(subset="title").head(top_k)


def naive_top_k(candidates: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Baseline: pure relevance ranking, no constraints."""
    return candidates.sort_values("relevance", ascending=False).head(top_k)
