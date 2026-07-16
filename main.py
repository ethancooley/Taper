"""Taper API: responsible workout recommendations.

Serves a single-page frontend and a /recommend endpoint. Users submit an
intake form (cold start) or pick a demo athlete profile (warm start); the
response includes both naive and constrained recommendations side by side,
plus the ACWR each pick would produce, so the tradeoff is visible in the UI.

Run locally:  uvicorn main:app --reload

Attribution: This file was written with the assistance of Claude (Anthropic).
"""

import os
import sys

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from model import (AthleteState, ResponsibleReranker, WorkoutRecommender,
                   naive_top_k)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

app = FastAPI(title="Taper")

catalog = pd.read_csv(os.path.join(DATA_DIR, "workouts.csv"))
interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
recommender = WorkoutRecommender(catalog).fit(interactions)
reranker = ResponsibleReranker()


class IntakeRequest(BaseModel):
    """Cold-start intake, or a demo athlete_id for warm start."""
    sports: list[str] = ["run"]
    max_intensity: int = 5
    weekly_hours: float = 5.0
    recent_weekly_hours: float = 2.0  # how much they've *actually* been doing
    athlete_id: int | None = None


def _state_from_hours(recent_weekly_hours: float,
                      weekly_hours: float) -> AthleteState:
    """Approximate 28-day load state from self-reported training volume."""
    state = AthleteState()
    # Assume moderate intensity (load ~70/hr) spread over the last 4 weeks.
    daily_load = (recent_weekly_hours * 70) / 7.0
    for _ in range(28):
        state.record(daily_load, "unknown")
    return state


def _workout_payload(rows: pd.DataFrame, state: AthleteState) -> list[dict]:
    out = []
    for _, r in rows.iterrows():
        out.append({
            "title": r.title,
            "sport": r.sport,
            "workout_type": r.workout_type,
            "intensity": int(r.intensity),
            "duration_min": int(r.duration_min),
            "load": float(r.load),
            "acwr_after": round(state.acwr_if_added(r.load), 2),
        })
    return out


@app.post("/recommend")
def recommend(req: IntakeRequest) -> dict:
    if req.athlete_id is not None:
        history = interactions[interactions.athlete_id == req.athlete_id]
        scored = recommender.score(history_ids=list(history.workout_id))
        state = AthleteState()
        merged = history.merge(catalog, on="workout_id")
        for _, row in merged.sort_values(["week", "day"]).iterrows():
            state.record(row.load, row.workout_type)
    else:
        scored = recommender.score(intake={
            "sports": req.sports, "max_intensity": req.max_intensity})
        state = _state_from_hours(req.recent_weekly_hours, req.weekly_hours)

    naive = naive_top_k(scored, top_k=5)
    safe = reranker.rerank(scored, state, top_k=5)
    return {
        "current_acwr": round(state.acute_load() / max(state.chronic_load(), 1e-9), 2),
        "naive": _workout_payload(naive, state),
        "constrained": _workout_payload(safe, state),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")),
          name="static")
