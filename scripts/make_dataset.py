"""Generate the synthetic dataset for Taper.

Creates:
  data/raw/workouts.csv      - workout catalog with content features
  data/raw/interactions.csv  - simulated athlete completion history (biased)

The interaction data is intentionally biased: athletes over-select the workout
types they already like and under-select recovery and strength work. A naive
recommender will learn and amplify this bias, which is the point of the demo.

Attribution: This file was written with the assistance of Claude (Anthropic).
"""

import os
import random

import numpy as np
import pandas as pd

SEED = 42

SPORTS = ["run", "bike", "swim", "strength"]

# (workout_type, intensity 1-5, typical duration range in minutes)
WORKOUT_TEMPLATES = {
    "run": [
        ("easy run", 2, (30, 60)),
        ("long run", 3, (75, 150)),
        ("tempo run", 4, (40, 70)),
        ("interval session", 5, (40, 70)),
        ("recovery jog", 1, (20, 40)),
    ],
    "bike": [
        ("endurance ride", 2, (60, 150)),
        ("tempo ride", 4, (60, 90)),
        ("interval ride", 5, (45, 75)),
        ("recovery spin", 1, (30, 45)),
    ],
    "swim": [
        ("technique swim", 2, (30, 50)),
        ("endurance swim", 3, (45, 70)),
        ("interval swim", 4, (40, 60)),
    ],
    "strength": [
        ("core session", 2, (20, 40)),
        ("full body strength", 3, (30, 60)),
        ("mobility session", 1, (20, 35)),
    ],
}

# Archetype: (name, sport prefs, weekly hours, preference for high intensity,
#             probability of skipping low-intensity/strength work)
ARCHETYPES = [
    ("eager_runner", {"run": 0.85, "strength": 0.10, "bike": 0.05},
     6, 0.75, 0.85),
    ("triathlete", {"run": 0.35, "bike": 0.40, "swim": 0.20, "strength": 0.05},
     9, 0.55, 0.65),
    ("comeback_athlete", {"run": 0.60, "bike": 0.25, "strength": 0.15},
     4, 0.40, 0.50),
]


def _load_score(intensity: int, duration_min: float) -> float:
    """Approximate a TSS-style training load score from intensity and duration."""
    intensity_factor = 0.45 + 0.14 * intensity  # 0.59 .. 1.15
    return round((duration_min / 60.0) * (intensity_factor ** 2) * 100, 1)


def build_workout_catalog(n_per_template: int = 4) -> pd.DataFrame:
    """Build a catalog of workouts by sampling durations from each template."""
    rng = random.Random(SEED)
    rows = []
    workout_id = 0
    for sport, templates in WORKOUT_TEMPLATES.items():
        for name, intensity, (dmin, dmax) in templates:
            for _ in range(n_per_template):
                duration = rng.randrange(dmin, dmax + 1, 5)
                rows.append({
                    "workout_id": workout_id,
                    "sport": sport,
                    "workout_type": name,
                    "intensity": intensity,
                    "duration_min": duration,
                    "load": _load_score(intensity, duration),
                    "title": f"{name.title()} - {duration}min",
                })
                workout_id += 1
    return pd.DataFrame(rows)


def simulate_interactions(catalog: pd.DataFrame, n_athletes: int = 50,
                          weeks: int = 8) -> pd.DataFrame:
    """Simulate biased completion histories for a population of athletes."""
    rng = np.random.default_rng(SEED)
    rows = []
    for athlete_id in range(n_athletes):
        arch_name, sport_prefs, hours, hi_pref, skip_easy = \
            ARCHETYPES[athlete_id % len(ARCHETYPES)]
        weekly_minutes = hours * 60
        for week in range(weeks):
            minutes_left = weekly_minutes * rng.uniform(0.8, 1.2)
            day = 0
            while minutes_left > 20 and day < 7:
                sport = rng.choice(list(sport_prefs), p=list(sport_prefs.values()))
                options = catalog[catalog.sport == sport]
                # Bias toward high intensity, away from easy/recovery work.
                weights = np.where(options.intensity >= 4, hi_pref, 1 - hi_pref)
                weights = np.where(options.intensity <= 1,
                                   weights * (1 - skip_easy), weights)
                weights = weights / weights.sum()
                w = options.sample(1, weights=weights, random_state=rng.integers(1e9)).iloc[0]
                rows.append({
                    "athlete_id": athlete_id,
                    "archetype": arch_name,
                    "week": week,
                    "day": day,
                    "workout_id": int(w.workout_id),
                })
                minutes_left -= w.duration_min
                day += rng.integers(1, 3)
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    catalog = build_workout_catalog()
    interactions = simulate_interactions(catalog)
    catalog.to_csv(os.path.join(out_dir, "workouts.csv"), index=False)
    interactions.to_csv(os.path.join(out_dir, "interactions.csv"), index=False)
    print(f"Wrote {len(catalog)} workouts and {len(interactions)} interactions.")


if __name__ == "__main__":
    main()
