"""
Steinhoff International — Questions 4.3 and 4.5
Counterfactual forecasting model and valuation.

4.3 asks how the company's position may have evolved had the scandal not
occurred. The model therefore starts from the last pre-disclosure balance sheet
date and projects five years forward on stated assumptions, with no scandal
effect applied. The projection is compared against what was actually reported.

Every assumption is a parameter, none is hard-coded into the arithmetic, and all
of them surface in the dashboard sidebar so that an assessor can vary them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FY = 2016            # last pre-disclosure reporting date in the captured data
FORECAST_YEARS = [2017, 2018, 2019, 2020, 2021]

SCENARIOS: dict[str, dict] = {
    "Conservative": dict(
        rev_growth=[0.04, 0.04, 0.03, 0.03, 0.03],
        op_margin=[0.070, 0.068, 0.066, 0.065, 0.065],
        da_pct=0.030, capex_pct=0.035, nwc_pct=0.010,
        wacc=0.135, g=0.020,
    ),
    "Base": dict(
        rev_growth=[0.08, 0.07, 0.06, 0.05, 0.05],
        op_margin=[0.090, 0.090, 0.088, 0.086, 0.085],
        da_pct=0.030, capex_pct=0.040, nwc_pct=0.012,
        wacc=0.120, g=0.025,
    ),
    "Expansionary": dict(
        rev_growth=[0.12, 0.11, 0.09, 0.08, 0.07],
        op_margin=[0.105, 0.105, 0.103, 0.100, 0.098],
        da_pct=0.030, capex_pct=0.050, nwc_pct=0.015,
        wacc=0.110, g=0.030,
    ),
}


# ----------------------------------------------------------------------------
# Base year
# ----------------------------------------------------------------------------

def base_position(ratios: pd.DataFrame, statements: pd.DataFrame,
                  base_fy: int = BASE_FY) -> dict:
    """Pull the starting position out of the captured data, not from constants."""
    p = statements.pivot_table(index="fy", columns="line_item", values="value",
                               aggfunc="first")

    def g(col, fy=base_fy):
        if col not in p.columns or fy not in p.index:
            return None
        v = p.at[fy, col]
        return None if pd.isna(v) else float(v)

    debt = (g("Interest-bearing borrowings (non-current)") or 0.0) + \
           (g("Interest-bearing borrowings (current)") or 0.0)
    cash = g("Cash and cash equivalents") or 0.0
    rev = g("Revenue")
    ta = float(ratios.at[base_fy, "total_assets"]) if base_fy in ratios.index else None

    return dict(
        fy=base_fy,
        currency="ZAR",
        revenue=rev,
        operating_profit=g("Operating profit"),
        total_assets=ta,
        total_equity=g("Total equity"),
        debt=debt,
        cash=cash,
        net_debt=debt - cash,
    )


# ----------------------------------------------------------------------------
# 4.3 — five year projection
# ----------------------------------------------------------------------------

def project(base: dict, scenario: str, tax_rate: float = 0.28,
            overrides: dict | None = None) -> pd.DataFrame:
    s = dict(SCENARIOS[scenario])
    if overrides:
        s.update({k: v for k, v in overrides.items() if v is not None})

    rev = base["revenue"]
    assets = base["total_assets"]
    equity = base["total_equity"]
    debt = base["debt"]
    cash = base["cash"]

    rows = []
    for i, fy in enumerate(FORECAST_YEARS):
        growth = s["rev_growth"][i]
        margin = s["op_margin"][i]

        rev = rev * (1 + growth)
        ebit = rev * margin
        da = rev * s["da_pct"]
        ebitda = ebit + da
        capex = rev * s["capex_pct"]
        d_nwc = rev * growth * s["nwc_pct"]
        tax = max(ebit, 0) * tax_rate
        nopat = ebit - tax
        fcff = nopat + da - capex - d_nwc

        # simplified statement roll-forward
        assets = assets + capex - da + d_nwc
        equity = equity + nopat            # no distribution assumed
        cash = cash + fcff
        rows.append(dict(
            fy=fy, revenue=rev, ebitda=ebitda, ebit=ebit, da=da, capex=capex,
            change_nwc=d_nwc, tax=tax, nopat=nopat, fcff=fcff,
            total_assets=assets, total_equity=equity, debt=debt, cash=cash,
            net_debt=debt - cash, scenario=scenario,
        ))
    return pd.DataFrame(rows).set_index("fy", drop=False)


def counterfactual_gap(proj: pd.DataFrame, ratios: pd.DataFrame,
                       zar_per_eur: float) -> pd.DataFrame:
    """Projected against reported, with the euro years converted at a stated rate."""
    rows = []
    for fy in proj.index:
        if fy not in ratios.index:
            continue
        actual_rev = ratios.at[fy, "revenue"]
        actual_eq = ratios.at[fy, "total_equity"]
        cur = ratios.at[fy, "currency"]
        k = zar_per_eur if cur == "EUR" else 1.0
        rows.append(dict(
            fy=fy,
            projected_revenue=proj.at[fy, "revenue"],
            actual_revenue=None if pd.isna(actual_rev) else actual_rev * k,
            projected_equity=proj.at[fy, "total_equity"],
            actual_equity=None if pd.isna(actual_eq) else actual_eq * k,
            reported_currency=cur,
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["revenue_gap"] = df["projected_revenue"] - df["actual_revenue"]
    df["equity_gap"] = df["projected_equity"] - df["actual_equity"]
    return df.set_index("fy", drop=False)


# ----------------------------------------------------------------------------
# 4.5 — valuation
# ----------------------------------------------------------------------------

def cost_of_equity(risk_free: float, beta: float, erp: float) -> float:
    return risk_free + beta * erp


def wacc(ke: float, kd: float, tax_rate: float, equity_weight: float) -> float:
    ew = min(max(equity_weight, 0.0), 1.0)
    return ke * ew + kd * (1 - tax_rate) * (1 - ew)


def dcf(proj: pd.DataFrame, discount_rate: float, terminal_growth: float,
        net_debt: float, shares_m: float) -> dict:
    if discount_rate <= terminal_growth:
        discount_rate = terminal_growth + 0.005

    fcff = proj["fcff"].values.astype(float)
    disc = [(1 + discount_rate) ** (i + 1) for i in range(len(fcff))]
    pv = [f / d for f, d in zip(fcff, disc)]

    tv = fcff[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_tv = tv / disc[-1]

    ev = float(np.sum(pv) + pv_tv)
    eq = ev - net_debt
    return dict(
        pv_explicit=float(np.sum(pv)), terminal_value=float(tv), pv_terminal=float(pv_tv),
        enterprise_value=ev, equity_value=eq,
        value_per_share=(eq / shares_m) if shares_m else float("nan"),
        discount_rate=discount_rate, terminal_growth=terminal_growth,
        pv_by_year=pd.Series(pv, index=proj.index),
    )


def sensitivity(proj: pd.DataFrame, net_debt: float, shares_m: float,
                waccs: list[float], growths: list[float]) -> pd.DataFrame:
    grid = pd.DataFrame(index=[f"{w:.1%}" for w in waccs],
                        columns=[f"{g:.1%}" for g in growths], dtype=float)
    for w in waccs:
        for g in growths:
            grid.at[f"{w:.1%}", f"{g:.1%}"] = dcf(proj, w, g, net_debt, shares_m)["value_per_share"]
    return grid


def ddm(dps: float, growth: float, ke: float) -> float | None:
    """Gordon growth. Returns None where it is not meaningful."""
    if dps is None or dps <= 0 or ke <= growth:
        return None
    return dps * (1 + growth) / (ke - growth)


def relative_value(proj: pd.DataFrame, base: dict, peer_pe: float, peer_ev_ebitda: float,
                   shares_m: float, net_debt: float, tax_rate: float = 0.28) -> pd.DataFrame:
    """Multiples applied to the first projected year."""
    fy = proj.index[0]
    ebitda = float(proj.at[fy, "ebitda"])
    earnings = float(proj.at[fy, "nopat"])
    rows = []
    if peer_pe and shares_m:
        eq = earnings * peer_pe
        rows.append(dict(method=f"P/E multiple ({peer_pe:.1f}x)", equity_value=eq,
                         value_per_share=eq / shares_m))
    if peer_ev_ebitda and shares_m:
        ev = ebitda * peer_ev_ebitda
        eq = ev - net_debt
        rows.append(dict(method=f"EV/EBITDA multiple ({peer_ev_ebitda:.1f}x)",
                         equity_value=eq, value_per_share=eq / shares_m))
    return pd.DataFrame(rows)
