"""
prediction_engine.py
====================
Machine-learning traffic prediction module.

Uses Scikit-Learn to:
  - Train a polynomial regression model on historical vehicle counts.
  - Predict vehicle counts 5 and 15 minutes into the future.
  - Classify the congestion trend (IMPROVING / STABLE / WORSENING).
  - Return full prediction history for dashboard charts.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# SCENARIO MULTIPLIERS
# ─────────────────────────────────────────────
SCENARIO_MULTIPLIERS = {
    "Normal":           1.0,
    "Rain Mode":        1.3,
    "Accident Mode":    1.8,
    "Road Block":       2.2,
    "Festival Traffic": 2.5,
    "School Rush Hour": 1.6,
}


# ─────────────────────────────────────────────
# PREDICTION ENGINE CLASS
# ─────────────────────────────────────────────

class TrafficPredictor:
    """
    Wraps a scikit-learn polynomial regression pipeline.
    Trains on historical vehicle-count data and predicts
    future counts at configurable horizons.
    """

    def __init__(self, degree: int = 3):
        self.degree = degree
        self.pipeline = Pipeline([
            ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ])
        self.trained = False
        self.n_samples = 0

    # --------------------------------------------------
    def fit(self, vehicle_counts: list):
        """Train the model on a sequence of vehicle counts."""
        counts = np.array(vehicle_counts, dtype=float)
        n = len(counts)

        if n < 5:
            self.trained = False
            return self

        X = np.arange(n).reshape(-1, 1)
        y = counts
        self.pipeline.fit(X, y)
        self.trained = True
        self.n_samples = n
        return self

    # --------------------------------------------------
    def predict_future(
        self,
        steps_5min: int = 5,
        steps_15min: int = 15,
        scenario: str = "Normal",
    ) -> dict:
        """
        Predict vehicle counts at 5-min and 15-min horizons.

        Returns
        -------
        dict with keys:
          pred_5min, pred_15min, trend, confidence, series
        """
        multiplier = SCENARIO_MULTIPLIERS.get(scenario, 1.0)

        if not self.trained:
            # Fallback: return last observed value
            return {
                "pred_5min":  0,
                "pred_15min": 0,
                "trend":      "STABLE",
                "confidence": 0,
                "series":     [],
            }

        n = self.n_samples
        future_5  = np.array([[n + steps_5min]])
        future_15 = np.array([[n + steps_15min]])

        raw_5  = float(self.pipeline.predict(future_5)[0])
        raw_15 = float(self.pipeline.predict(future_15)[0])

        pred_5  = max(0, int(raw_5  * multiplier))
        pred_15 = max(0, int(raw_15 * multiplier))

        # Build a 30-step forecast series for chart display
        future_idx = np.arange(n, n + 31).reshape(-1, 1)
        series_raw = self.pipeline.predict(future_idx) * multiplier
        series = [max(0, int(v)) for v in series_raw]

        # Trend classification
        current = float(self.pipeline.predict(np.array([[n - 1]]))[0])
        if pred_15 > current * 1.1:
            trend = "WORSENING"
        elif pred_15 < current * 0.9:
            trend = "IMPROVING"
        else:
            trend = "STABLE"

        # Rough confidence: inverse of coefficient of variation
        confidence = min(99, max(10, int(100 - abs(raw_15 - raw_5) / max(1, raw_5) * 50)))

        return {
            "pred_5min":  pred_5,
            "pred_15min": pred_15,
            "trend":      trend,
            "confidence": confidence,
            "series":     series,
        }

    # --------------------------------------------------
    def mae_score(self, vehicle_counts: list) -> float:
        """Return in-sample MAE as a model quality indicator."""
        if not self.trained or len(vehicle_counts) < 5:
            return float("nan")
        counts = np.array(vehicle_counts, dtype=float)
        n = len(counts)
        X = np.arange(n).reshape(-1, 1)
        y_pred = self.pipeline.predict(X)
        return round(mean_absolute_error(counts, y_pred), 2)


# ─────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE
# ─────────────────────────────────────────────

_predictor = TrafficPredictor(degree=3)


def train_and_predict(df: pd.DataFrame, scenario: str = "Normal") -> dict:
    """
    Train on dataframe['vehicle_count'] and return predictions.

    Parameters
    ----------
    df       : pandas DataFrame with a 'vehicle_count' column
    scenario : one of SCENARIO_MULTIPLIERS keys

    Returns
    -------
    dict from TrafficPredictor.predict_future()
    """
    if df is None or len(df) < 3:
        return {
            "pred_5min":  0,
            "pred_15min": 0,
            "trend":      "STABLE",
            "confidence": 0,
            "series":     [],
        }

    counts = df["vehicle_count"].tolist()
    _predictor.fit(counts)
    result = _predictor.predict_future(scenario=scenario)
    result["mae"] = _predictor.mae_score(counts)
    return result


def build_forecast_df(current_count: int, series: list) -> pd.DataFrame:
    """
    Build a small DataFrame for chart display.
    Combines the current count with the forecast series.
    """
    if not series:
        return pd.DataFrame({"Minute": [0], "Predicted Vehicles": [current_count]})

    minutes = list(range(0, len(series) + 1))
    values  = [current_count] + series
    return pd.DataFrame({"Minute": minutes, "Predicted Vehicles": values})
