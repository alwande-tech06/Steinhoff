"""
Steinhoff International — Question 4.4
Two machine learning models: Linear Regression and Decision Tree.

The module runs in one of two modes, decided by whether a share price file is
present in ./data/share_price.csv:

  PRICE MODE     Target is the next-day closing price. Features are lagged
                 returns, moving averages, realised volatility and volume.
                 This is the mode the assessment brief asks for.

  DISTRESS MODE  Fallback used while the price series is still being sourced.
                 Both models are trained on a simulated population of company
                 years to predict a continuous financial distress index from
                 ratio features, and Steinhoff's own years are then scored
                 against that model. Clearly labelled as a substitute in the
                 dashboard, because it does not answer 4.4 as written.

Both modes evaluate with the same measures (R squared, RMSE, MAE, MAPE) so the
two models remain directly comparable, which is what 4.4 requires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, export_text

RANDOM_STATE = 42

FEATURE_DEFS_DISTRESS = {
    "return_on_assets": "profit for the year / total assets",
    "debt_ratio": "total liabilities / total assets",
    "current_ratio": "current assets / current liabilities",
    "operating_margin": "operating profit / revenue",
    "goodwill_pct_assets": "goodwill / total assets",
    "soft_assets_pct": "(goodwill + intangibles) / total assets",
    "asset_turnover": "revenue / total assets",
}

FEATURE_DEFS_PRICE = {
    "lag_1": "closing price, one trading day earlier",
    "lag_5": "closing price, five trading days earlier",
    "ma_5": "five day moving average of the close",
    "ma_20": "twenty day moving average of the close",
    "ret_1": "one day return",
    "volatility_20": "twenty day standard deviation of daily returns",
    "volume_rel": "volume divided by its twenty day average",
}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nz = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else float("nan")
    return {
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "n_test": int(len(y_true)),
    }


def _fit_pair(X_tr, y_tr, X_te, y_te, feature_names, max_depth=4):
    """Fit both required models and return everything the dashboard needs."""
    lin = LinearRegression().fit(X_tr, y_tr)
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=8,
                                 random_state=RANDOM_STATE).fit(X_tr, y_tr)

    p_lin_te, p_tree_te = lin.predict(X_te), tree.predict(X_te)
    p_lin_tr, p_tree_tr = lin.predict(X_tr), tree.predict(X_tr)

    return {
        "models": {"Linear Regression": lin, "Decision Tree": tree},
        "metrics": {
            "Linear Regression": {"test": _metrics(y_te, p_lin_te),
                                  "train": _metrics(y_tr, p_lin_tr)},
            "Decision Tree": {"test": _metrics(y_te, p_tree_te),
                              "train": _metrics(y_tr, p_tree_tr)},
        },
        "predictions": {"Linear Regression": p_lin_te, "Decision Tree": p_tree_te},
        "actual": np.asarray(y_te, dtype=float),
        "coefficients": pd.DataFrame({
            "feature": feature_names,
            "coefficient": lin.coef_,
        }).sort_values("coefficient", key=abs, ascending=False),
        "importance": pd.DataFrame({
            "feature": feature_names,
            "importance": tree.feature_importances_,
        }).sort_values("importance", ascending=False),
        "tree_rules": export_text(tree, feature_names=list(feature_names), decimals=3),
    }


# ----------------------------------------------------------------------------
# Price mode
# ----------------------------------------------------------------------------

def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values("date").reset_index(drop=True).copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df["lag_1"] = df["close"].shift(1)
    df["lag_5"] = df["close"].shift(5)
    df["ma_5"] = df["close"].rolling(5).mean().shift(1)
    df["ma_20"] = df["close"].rolling(20).mean().shift(1)
    df["ret_1"] = df["close"].pct_change().shift(1)
    df["volatility_20"] = df["close"].pct_change().rolling(20).std().shift(1)
    vol_ma = df["volume"].rolling(20).mean().shift(1)
    df["volume_rel"] = (df["volume"].shift(1) / vol_ma).replace([np.inf, -np.inf], np.nan)
    df["volume_rel"] = df["volume_rel"].fillna(1.0)
    df["target"] = df["close"]
    return df.dropna(subset=list(FEATURE_DEFS_PRICE) + ["target"]).reset_index(drop=True)


def run_price_mode(prices: pd.DataFrame, split_date: str | pd.Timestamp) -> dict:
    """Train before the structural break, test into it.

    Training across the collapse would let the model learn the crash it is meant
    to be tested on. Training only on the earlier period and testing into the
    collapse is the design that actually asks whether the market data carried a
    warning, and is stated as such in the report.
    """
    feats = build_price_features(prices)
    split = pd.Timestamp(split_date)
    names = list(FEATURE_DEFS_PRICE)

    tr = feats[feats["date"] < split]
    te = feats[feats["date"] >= split]
    if len(tr) < 60 or len(te) < 10:
        cut = int(len(feats) * 0.8)
        tr, te = feats.iloc[:cut], feats.iloc[cut:]

    out = _fit_pair(tr[names].values, tr["target"].values,
                    te[names].values, te["target"].values, names)
    out.update(mode="price", feature_defs=FEATURE_DEFS_PRICE,
               test_index=te["date"].values, train_rows=len(tr), test_rows=len(te),
               split=split, target_label="Closing price")
    return out


# ----------------------------------------------------------------------------
# Distress mode (fallback)
# ----------------------------------------------------------------------------

def _simulate_population(n: int = 4000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Simulated company years spanning healthy to severely distressed.

    Used only to fit the models when no share price series is available. The
    weights below encode the direction each indicator moves in distress; they
    are assumptions, not findings, and are disclosed as such.
    """
    rng = np.random.default_rng(seed)
    roa = rng.normal(0.05, 0.07, n)
    gearing = np.clip(rng.normal(0.55, 0.18, n), 0.05, 1.4)
    current = np.clip(rng.normal(1.35, 0.45, n), 0.15, 3.5)
    opm = rng.normal(0.07, 0.06, n)
    gw = np.clip(rng.normal(0.15, 0.12, n), 0.0, 0.65)
    soft = np.clip(gw + np.abs(rng.normal(0.10, 0.08, n)), 0.0, 0.9)
    turn = np.clip(rng.normal(0.85, 0.35, n), 0.05, 2.5)

    latent = (-6.0 * roa + 2.4 * (gearing - 0.5) - 1.1 * (current - 1.3)
              - 5.0 * opm + 1.8 * gw + 1.5 * (soft - 0.25) - 0.5 * (turn - 0.85)
              + rng.normal(0, 0.45, n))
    distress = 1 / (1 + np.exp(-latent))

    return pd.DataFrame({
        "return_on_assets": roa, "debt_ratio": gearing, "current_ratio": current,
        "operating_margin": opm, "goodwill_pct_assets": gw, "soft_assets_pct": soft,
        "asset_turnover": turn, "distress_index": distress,
    })


def run_distress_mode(ratios: pd.DataFrame, n: int = 4000) -> dict:
    pop = _simulate_population(n)
    names = list(FEATURE_DEFS_DISTRESS)
    cut = int(len(pop) * 0.75)
    tr, te = pop.iloc[:cut], pop.iloc[cut:]

    out = _fit_pair(tr[names].values, tr["distress_index"].values,
                    te[names].values, te["distress_index"].values, names)

    scored = ratios.copy()
    for c in names:
        if c not in scored.columns:
            scored[c] = np.nan
    usable = scored.dropna(subset=names)
    rows = []
    if not usable.empty:
        X = usable[names].values
        for label, model in out["models"].items():
            pred = np.clip(model.predict(X), 0, 1)
            for fy, v in zip(usable.index, pred):
                rows.append(dict(fy=int(fy), model=label, distress_index=float(v)))
    out["company_scores"] = pd.DataFrame(rows)

    out.update(mode="distress", feature_defs=FEATURE_DEFS_DISTRESS,
               train_rows=len(tr), test_rows=len(te),
               target_label="Financial distress index (0 = sound, 1 = severe)")
    return out


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def load_price_file(folder: str | Path = "data") -> pd.DataFrame | None:
    p = Path(folder) / "share_price.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    return df if len(df) >= 80 else None


def run(ratios: pd.DataFrame, folder: str | Path = "data",
        split_date: str = "2017-10-01") -> dict:
    prices = load_price_file(folder)
    if prices is not None:
        return run_price_mode(prices, split_date)
    return run_distress_mode(ratios)
