"""
Steinhoff International — Financial Intelligence Dashboard
Question 4: analytics, business intelligence, forecasting, machine learning, valuation.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import etl
import ml

# --------------------------------------------------------------------------
# Visual identity: a warm archive palette. Ink-brown ground, moss for neutral
# series, bronze for secondary, mustard and oxblood reserved strictly for
# deterioration, so colour always carries a finding.
# --------------------------------------------------------------------------
INK = "#17130F"
SURFACE = "#221C17"
LINE = "#3A2F26"
TEXT = "#F2EAE1"
MUTED = "#A3948A"
MOSS = "#6F9E86"
BRONZE = "#C08552"
MUSTARD = "#D4A017"
OXBLOOD = "#A8434E"
PLUM = "#8C6A9B"
GOOD = "#5B9370"

PHASE_COLOURS = {"Before": MOSS, "During": MUSTARD, "After": OXBLOOD}
STATUS_COLOUR = {"Healthy": GOOD, "Warning": MUSTARD, "Critical": OXBLOOD, "No data": "#4A3F36"}
ZONE_COLOUR = {"Safe": GOOD, "Grey": MUSTARD, "Distress": OXBLOOD}

DATA_FILE = "Steinhoff Data-Version 2 2026.xlsx"

st.set_page_config(page_title="Steinhoff Financial Intelligence Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp {{ background: {INK}; color: {TEXT}; font-family: 'IBM Plex Sans', sans-serif; }}
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
section[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"],
section[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"] div[data-baseweb="base-input"] {{
  background: {INK} !important; border: 1px solid {LINE} !important; }}
section[data-testid="stSidebar"] input[data-testid="stNumberInputField"] {{
  background: transparent !important; color: {TEXT} !important;
  -webkit-text-fill-color: {TEXT} !important; }}
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"],
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"] {{
  background: {INK} !important; border-color: {LINE} !important; }}
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"] svg,
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"] svg {{
  fill: {TEXT} !important; }}
h1, h2, h3, h4 {{ font-family: 'Spectral', serif; color: {TEXT}; font-weight: 600;
  letter-spacing: -.01em; }}
h1 {{ font-size: 2.15rem; margin-bottom: .1rem; }}
h3 {{ font-size: 1.2rem; margin-top: .3rem; }}
.block-container {{ padding-top: 2rem; }}

.masthead {{ border-bottom: 1px solid {LINE}; padding-bottom: .75rem; margin-bottom: 1rem; }}
.masthead .sub {{ font-family: 'IBM Plex Mono', monospace; font-size: .72rem;
  color: {MUTED}; letter-spacing: .11em; text-transform: uppercase; }}

.card {{ background: {SURFACE}; border: 1px solid {LINE};
  border-left: 3px solid var(--tone, {MOSS}); border-radius: 2px;
  padding: .7rem .85rem; height: 100%; }}
.card .label {{ font-family: 'IBM Plex Mono', monospace; text-transform: uppercase;
  letter-spacing: .09em; font-size: .66rem; color: {MUTED}; }}
.card .value {{ font-family: 'Spectral', serif; font-size: 1.6rem; font-weight: 600;
  color: {TEXT}; line-height: 1.45; }}
.card .delta {{ font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: {MUTED}; }}

.note {{ background: {SURFACE}; border-left: 3px solid {MUSTARD}; padding: .6rem .85rem;
  font-size: .86rem; color: {MUTED}; margin: .5rem 0 1rem 0; }}
.note b {{ color: {TEXT}; }}
.crit {{ border-left-color: {OXBLOOD}; }}

table.reg {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
table.reg th {{ font-family: 'IBM Plex Mono', monospace; font-size: .64rem;
  letter-spacing: .1em; text-transform: uppercase; color: {MUTED}; text-align: left;
  padding: .45rem .55rem; border-bottom: 1px solid {LINE}; }}
table.reg td {{ padding: .5rem .55rem; border-bottom: 1px solid {LINE}; vertical-align: top; }}
table.reg td.per {{ font-family: 'IBM Plex Mono', monospace; color: {BRONZE}; white-space: nowrap; }}
.stDataFrame {{ border: 1px solid {LINE}; }}
footer, #MainMenu {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: {INK}; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load(path: str):
    return etl.load_all(path)


@st.cache_data(show_spinner=False)
def run_ml(_ratios: pd.DataFrame, folder: str):
    return ml.run(_ratios, folder=folder)


if not Path(DATA_FILE).exists():
    st.error(f"Capture workbook not found: {DATA_FILE}")
    st.stop()

statements, ratios, checks, events, scorecard, prices, distress, beneish = load(DATA_FILE)
base = an.base_position(ratios, statements)
YEARS = sorted(ratios.index)

st.markdown(f"""
<div class="masthead">
  <h1>Steinhoff International</h1>
  <div class="sub">Financial analytics, business intelligence,
  forecasting, machine learning and valuation &nbsp;·&nbsp; FY2012&ndash;FY2018</div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar — every assumption is a parameter
# --------------------------------------------------------------------------

sb = st.sidebar
sb.markdown("### Filters")
sb.caption("These affect the historical tabs. Forecast, valuation and "
           "predictive analytics look forward from a fixed base year "
           "and are unaffected.")
year_lo, year_hi = sb.slider("Financial years", YEARS[0], YEARS[-1],
                              (YEARS[0], YEARS[-1]))

sb.markdown("**Scandal timeline**")
pre_scandal_end = sb.slider("Pre-scandal period ends", YEARS[0], YEARS[-1] - 1, 2016)
scandal_end = sb.slider("Scandal period ends", pre_scandal_end + 1, YEARS[-1],
                         max(2017, pre_scandal_end + 1))
sb.markdown("---")

PHASE_MAP = etl.phase_map(YEARS, pre_scandal_end, scandal_end)
YEARS_f = [y for y in YEARS if year_lo <= y <= year_hi]
ratios_f = ratios.loc[YEARS_f]
statements_f = statements[statements.fy.between(year_lo, year_hi)]
scorecard_f = scorecard[scorecard.fy.between(year_lo, year_hi)]
events_f = (events[events.fy_anchor.isna() | events.fy_anchor.between(year_lo, year_hi)]
            if not events.empty else events)
prices_f = prices[prices.date.dt.year.between(year_lo, year_hi)] if not prices.empty else prices
checks_f = checks[checks.fy.isna() | checks.fy.between(year_lo, year_hi)]
distress_f = distress[distress.fy.between(year_lo, year_hi)] if not distress.empty else distress
beneish_f = beneish[beneish.index.isin(YEARS_f)] if not beneish.empty else beneish

sb.markdown("### Assumptions")
sb.caption("Nothing below is a finding. Each is an input an assessor can vary.")

scenario = sb.selectbox("Forecast scenario", list(an.SCENARIOS), index=1)
tax_rate = sb.slider("Corporate tax rate", 0.15, 0.35, 0.28, 0.01)
# Version 2 restates every year in rand, so no conversion assumption is needed.
zar_per_eur = 1.0

sb.markdown("---")
sb.markdown("**Cost of capital**")
risk_free = sb.slider("Risk-free rate", 0.04, 0.14, 0.090, 0.005)
erp = sb.slider("Equity risk premium", 0.03, 0.10, 0.055, 0.005)
beta = sb.slider("Beta", 0.4, 2.5, 1.20, 0.05)
kd = sb.slider("Cost of debt", 0.04, 0.18, 0.095, 0.005)
eq_weight = sb.slider("Equity weight", 0.1, 1.0, 0.55, 0.05)
term_g = sb.slider("Terminal growth", 0.0, 0.06, 0.025, 0.005)

sb.markdown("---")
sb.markdown("**Share and peer inputs**")
shares_m = sb.number_input("Shares in issue (millions)", 100.0, 10000.0, 4300.0, 50.0)
peer_pe = sb.number_input("Peer P/E multiple", 0.0, 40.0, 14.0, 0.5)
peer_ev = sb.number_input("Peer EV/EBITDA multiple", 0.0, 30.0, 9.0, 0.5)
dps = sb.number_input("Dividend per share (base year)", 0.0, 50.0, 0.0, 0.05)

ke = an.cost_of_equity(risk_free, beta, erp)
wacc_v = an.wacc(ke, kd, tax_rate, eq_weight)
sb.markdown(f"<div class='note'>Cost of equity <b>{ke:.2%}</b><br>WACC <b>{wacc_v:.2%}</b></div>",
            unsafe_allow_html=True)

proj = an.project(base, scenario, tax_rate)
val = an.dcf(proj, wacc_v, term_g, base["net_debt"], shares_m)
gap = an.counterfactual_gap(proj, ratios, zar_per_eur)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def kpi(container, label, value, delta="", tone=MOSS):
    container.markdown(
        f"<div class='card' style='--tone:{tone}'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div><div class='delta'>{delta}</div></div>",
        unsafe_allow_html=True)


def rm(v, dec=0):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:,.{dec}f}"


def pct(v, dec=1):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v*100:,.{dec}f}%"


def xx(v, dec=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:,.{dec}f}x"


def style(fig, height=380, ytick=None, legend=False):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=34, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=TEXT),
        showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=INK, font_size=12, font_family="IBM Plex Sans"),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="outside", tickcolor=LINE,
                     tickfont=dict(family="IBM Plex Mono", size=11, color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=True, zerolinecolor=LINE,
                     linecolor=SURFACE, tickformat=ytick,
                     tickfont=dict(family="IBM Plex Mono", size=11, color=MUTED))
    return fig


def show(fig, height=380, ytick=None, legend=False):
    st.plotly_chart(style(fig, height, ytick, legend), use_container_width=True,
                    config={"displayModeBar": False})


def phase_bands(fig):
    for ph, colour in PHASE_COLOURS.items():
        yrs = [y for y, p in PHASE_MAP.items() if p == ph]
        if not yrs:
            continue
        fig.add_vrect(x0=min(yrs) - .5, x1=max(yrs) + .5, fillcolor=colour,
                      opacity=.07, line_width=0, layer="below")
    return fig


tabs = st.tabs(["Overview", "Profitability", "Financial position", "Earnings quality",
                "Red flags & chronology", "Forecast", "Valuation",
                "Predictive analytics", "Data & integrity"])


# --------------------------------------------------------------------------
# TAB 1 — Overview
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Where the company stood, and when it turned")
    last = YEARS_f[-1]
    cards = st.columns(6)
    kpi(cards[0], f"FY{last} current ratio", xx(ratios.at[last, "current_ratio"]),
        f"FY2013: {xx(ratios.at[2013, 'current_ratio'])}",
        OXBLOOD if (ratios.at[last, "current_ratio"] or 0) < 1 else MOSS)
    kpi(cards[1], f"FY{last} debt to equity", xx(ratios.at[last, "debt_to_equity"]),
        f"FY2013: {xx(ratios.at[2013, 'debt_to_equity'])}", OXBLOOD)
    kpi(cards[2], f"FY{last} operating margin", pct(ratios.at[last, "operating_margin"]),
        f"FY2016: {pct(ratios.at[2016, 'operating_margin'])}", OXBLOOD)
    kpi(cards[3], "Goodwill share FY2016", pct(ratios.at[2016, "goodwill_pct_assets"]),
        "of total assets", MUSTARD)
    kpi(cards[4], f"FY{last} total equity", rm(ratios.at[last, "total_equity"]),
        "ZAR million", OXBLOOD)
    kpi(cards[5], "Critical findings", str(int((checks_f.severity == "Critical").sum())),
        "in the captured data", OXBLOOD)

    st.markdown(f"<div class='note'><b>Reading the colour.</b> Green marks FY{YEARS[0]}"
                f"&ndash;FY{pre_scandal_end} before disclosure, mustard FY{pre_scandal_end + 1}"
                f"&ndash;FY{scandal_end} the scandal period, oxblood FY{scandal_end + 1} onward "
                "the restated aftermath. Every year is stated in rand in this version of the "
                "workbook, so absolute values are comparable across the whole period.</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=ratios_f.index, y=ratios_f["revenue"], marker_color=MOSS, name="Revenue",
                    width=.6, hovertemplate="FY%{x}<br>R%{y:,.0f}m<extra></extra>")
        fig.add_scatter(x=ratios_f.index, y=ratios_f["profit"], mode="lines+markers", name="Profit",
                        line=dict(color=BRONZE, width=2, dash="dot"),
                        marker=dict(size=8, symbol="diamond", color=BRONZE),
                        hovertemplate="FY%{x}<br>%{y:,.0f}m<extra></extra>")
        phase_bands(fig)
        show(fig, 400, legend=True)
        st.caption("Revenue with profit for the year overlaid. Rand millions throughout.")
    with c2:
        fig = go.Figure()
        fig.add_bar(x=ratios_f.index, y=ratios_f["total_equity"], width=.6,
                    marker_color=[GOOD if (v or 0) > 5000 else OXBLOOD for v in ratios_f["total_equity"]],
                    hovertemplate="FY%{x}<br>Equity %{y:,.0f}m<extra></extra>")
        phase_bands(fig)
        show(fig, 400)
        st.caption("Total equity — the buffer available to absorb a write-down.")


# --------------------------------------------------------------------------
# TAB 2 — Profitability
# --------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### Margins hold up until the year of disclosure")
    st.caption("Margins are proportions, so they remain comparable across the currency break.")

    cols = st.columns(3)
    for col, (lab, key) in zip(cols, [("Gross margin", "gross_margin"),
                                      ("Operating margin", "operating_margin"),
                                      ("Net margin", "net_margin")]):
        with col:
            ser = ratios_f[key].dropna()
            fig = go.Figure()
            fig.add_bar(x=ser.index, y=ser.values, width=.6,
                        marker_color=[OXBLOOD if v < 0 else MOSS for v in ser.values],
                        hovertemplate="FY%{x}<br>%{y:.1%}<extra></extra>")
            phase_bands(fig)
            fig.update_layout(title=dict(text=lab, font=dict(family="Spectral", size=15), x=0))
            show(fig, 300, ytick=".0%")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for lab, key, colour in [("Return on assets", "return_on_assets", MOSS),
                                 ("Return on equity", "return_on_equity", BRONZE)]:
            ser = ratios_f[key].dropna()
            fig.add_scatter(x=ser.index, y=ser.values, mode="lines+markers", name=lab,
                            line=dict(color=colour, width=2), marker=dict(size=8),
                            hovertemplate="FY%{x}<br>%{y:.1%}<extra></extra>")
        phase_bands(fig)
        show(fig, 340, ytick=".0%", legend=True)
        st.caption("Both use profit for the year, applied consistently.")
    with c2:
        ser = ratios_f["asset_turnover"].dropna()
        fig = go.Figure()
        fig.add_scatter(x=ser.index, y=ser.values, mode="lines+markers",
                        line=dict(color=PLUM, width=2), marker=dict(size=8),
                        hovertemplate="FY%{x}<br>%{y:.2f}x<extra></extra>")
        phase_bands(fig)
        show(fig, 340)
        st.caption("Asset turnover — revenue generated per unit of assets.")

    st.markdown("<div class='note crit'><b>Caution on FY2015.</b> Revenue falls by two thirds and "
                "recovers fully the following year. That is not a trading pattern, and every FY2015 "
                "margin inherits the distortion. The finding is recorded in the integrity register."
                "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# TAB 3 — Financial position
# --------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### Liquidity and gearing")

    cards = st.columns(4)
    kpi(cards[0], "Current ratio FY2016", xx(ratios.at[2016, "current_ratio"]), "", OXBLOOD)
    kpi(cards[1], "Quick ratio FY2016", xx(ratios.at[2016, "quick_ratio"]), "", OXBLOOD)
    kpi(cards[2], "Debt to equity FY2016", xx(ratios.at[2016, "debt_to_equity"]), "", MUSTARD)
    kpi(cards[3], "Debt ratio FY2016", pct(ratios.at[2016, "debt_ratio"]), "", MUSTARD)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for lab, key, colour in [("Current ratio", "current_ratio", MOSS),
                                 ("Quick ratio", "quick_ratio", BRONZE)]:
            ser = ratios_f[key].dropna()
            fig.add_scatter(x=ser.index, y=ser.values, mode="lines+markers", name=lab,
                            line=dict(color=colour, width=2), marker=dict(size=8),
                            hovertemplate="FY%{x}<br>%{y:.2f}x<extra></extra>")
        fig.add_hline(y=1.0, line=dict(color=OXBLOOD, width=1, dash="dash"))
        phase_bands(fig)
        show(fig, 340, legend=True)
        st.caption("Below the dashed line, short-term obligations exceed short-term assets.")
    with c2:
        fig = go.Figure()
        ser = ratios_f["debt_to_equity"].dropna()
        fig.add_bar(x=ser.index, y=ser.values, width=.6,
                    marker_color=[OXBLOOD if v > 2 else (MUSTARD if v > 1 else MOSS)
                                  for v in ser.values],
                    hovertemplate="FY%{x}<br>%{y:.2f}x<extra></extra>")
        phase_bands(fig)
        show(fig, 340)
        st.caption("Debt to equity. Gearing rises as the equity base thins.")

    st.markdown("#### Asset composition")
    comp = statements_f[statements_f.line_item.isin(
        ["Goodwill", "Intangible assets", "Property, plant and equipment",
         "Inventories", "Cash and cash equivalents"])]
    if not comp.empty:
        piv = comp.pivot_table(index="fy", columns="line_item", values="value", aggfunc="first")
        fig = go.Figure()
        for col, colour in zip(piv.columns, [MUSTARD, BRONZE, MOSS, PLUM, GOOD]):
            fig.add_bar(x=piv.index, y=piv[col], name=col, marker_color=colour,
                        hovertemplate="FY%{x}<br>" + col + " %{y:,.0f}m<extra></extra>")
        fig.update_layout(barmode="stack")
        show(fig, 360, legend=True)
        st.caption("Rand millions. The goodwill and intangibles blocks carry the write-down exposure.")


# --------------------------------------------------------------------------
# TAB 4 — Earnings quality
# --------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### Soft assets and the quality of reported earnings")
    st.markdown("<div class='note'>Goodwill and intangibles are the assets whose value rests on "
                "management judgement rather than observable price. A rising share signals both "
                "acquisition-led growth and exposure to write-down.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for lab, key, colour in [("Goodwill share of assets", "goodwill_pct_assets", MUSTARD),
                                 ("Soft assets share", "soft_assets_pct", OXBLOOD)]:
            ser = ratios_f[key].dropna()
            fig.add_scatter(x=ser.index, y=ser.values, mode="lines+markers", name=lab,
                            line=dict(color=colour, width=2), marker=dict(size=8),
                            hovertemplate="FY%{x}<br>%{y:.1%}<extra></extra>")
        phase_bands(fig)
        show(fig, 360, ytick=".0%", legend=True)
    with c2:
        acc = ratios_f["accruals_ratio"].dropna()
        if acc.empty:
            st.markdown("<div class='note crit'><b>Accruals test unavailable.</b> The accruals "
                        "ratio compares profit with operating cash flow. Cash flow is captured for "
                        "FY2013 and FY2014 only, so the test cannot be run across the period. This "
                        "is a data limitation, not a finding about the company.</div>",
                        unsafe_allow_html=True)
        else:
            fig = go.Figure()
            fig.add_bar(x=acc.index, y=acc.values, width=.6, marker_color=BRONZE,
                        hovertemplate="FY%{x}<br>%{y:.1%}<extra></extra>")
            phase_bands(fig)
            show(fig, 360, ytick=".0%")
            st.caption("(Profit less operating cash flow) over total assets. "
                       "Persistently positive values mean profit is not converting to cash.")



    st.markdown("### Altman Z'' and manipulation indices")
    st.caption("Both panels are read from the workbook's Distress_Models sheet as captured, "
               "and are shown alongside the recomputed ratios rather than in place of them.")

    c1, c2 = st.columns(2)
    with c1:
        if distress_f.empty:
            st.info("No Z-score panel in the workbook.")
        else:
            fig = go.Figure()
            fig.add_hrect(y0=2.6, y1=max(4, float(distress_f["z_score"].max()) + 1),
                          fillcolor=GOOD, opacity=.10, line_width=0)
            fig.add_hrect(y0=1.1, y1=2.6, fillcolor=MUSTARD, opacity=.10, line_width=0)
            fig.add_hrect(y0=min(-7, float(distress_f["z_score"].min()) - 1), y1=1.1,
                          fillcolor=OXBLOOD, opacity=.10, line_width=0)
            fig.add_scatter(x=distress_f.index, y=distress_f["z_score"], mode="lines+markers+text",
                            line=dict(color=TEXT, width=2),
                            marker=dict(size=11, color=[ZONE_COLOUR.get(z, MUTED)
                                                        for z in distress_f["zone"]]),
                            text=distress_f["zone"], textposition="top center",
                            textfont=dict(size=10, color=MUTED),
                            hovertemplate="FY%{x}<br>Z'' %{y:.2f}<extra></extra>")
            show(fig, 360)
            st.caption("Above 2.6 safe, 1.1 to 2.6 grey, below 1.1 distress. "
                       "The company enters the distress zone in FY2015 — two years before disclosure.")
    with c2:
        if beneish_f.empty:
            st.info("No manipulation indices in the workbook.")
        else:
            b = beneish_f.dropna(how="all")
            fig = go.Figure(go.Heatmap(
                z=b.T.values, x=[f"FY{y}" for y in b.index], y=list(b.columns),
                colorscale=[[0, SURFACE], [0.5, MUSTARD], [1, OXBLOOD]], zmid=1.0,
                hovertemplate="%{y} %{x}<br>%{z:.3f}<extra></extra>",
                colorbar=dict(tickfont=dict(color=MUTED, size=10))))
            show(fig, 360)
            st.caption("Partial Beneish indices. A value above 1.0 points in the direction "
                       "consistent with manipulation for that index.")

    st.markdown("<div class='note'><b>Why the M-Score is partial.</b> The full eight-variable "
                "Beneish M-Score also needs a depreciation index, an SG&A index and total accruals. "
                "The summarised statements do not disaggregate depreciation or SG&A, so a defensible "
                "full M-Score cannot be computed from them. That limitation in disclosure granularity "
                "is itself worth reporting as a finding.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# TAB 5 — Red flags & chronology
# --------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### Indicator scorecard")
    st.caption("Green healthy, mustard warning, oxblood critical, dark grey not captured. "
               "Read down the columns: the warnings begin years before disclosure.")

    inds = [s[0] for s in etl.SCORECARD]
    fig = go.Figure()
    for _, r in scorecard_f.iterrows():
        y = inds.index(r["indicator"])
        fig.add_shape(type="rect", x0=r["fy"] - .45, x1=r["fy"] + .45,
                      y0=y - .4, y1=y + .4,
                      fillcolor=STATUS_COLOUR[r["status"]], line=dict(color=SURFACE, width=2))
    fig.add_scatter(x=scorecard_f["fy"], y=[inds.index(i) for i in scorecard_f["indicator"]],
                    mode="markers", marker=dict(size=1, color="rgba(0,0,0,0)"),
                    customdata=scorecard_f[["indicator", "status", "note"]].values,
                    hovertemplate="<b>%{customdata[0]}</b> FY%{x}<br>%{customdata[1]}"
                                  "<br>%{customdata[2]}<extra></extra>")
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(inds))), ticktext=inds,
                     autorange="reversed", showgrid=False, linecolor=SURFACE,
                     tickfont=dict(family="IBM Plex Sans", size=12, color=TEXT))
    fig.update_xaxes(tickmode="array", tickvals=YEARS_f, ticktext=[f"FY{y}" for y in YEARS_f],
                     showgrid=False, linecolor=SURFACE)
    show(fig, 70 + 42 * len(inds))

    counts = scorecard_f.groupby(["fy", "status"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for status in ["Healthy", "Warning", "Critical", "No data"]:
        if status in counts.columns:
            fig.add_bar(x=counts.index, y=counts[status], name=status,
                        marker_color=STATUS_COLOUR[status],
                        hovertemplate="FY%{x}<br>" + status + ": %{y}<extra></extra>")
    fig.update_layout(barmode="stack")
    show(fig, 260, legend=True)
    st.caption("Count of indicators at each status, year by year.")

    st.markdown("### Share price anchor points")
    if prices_f.empty:
        st.info("No share price observations captured.")
    else:
        exact = prices_f[prices_f["exact"]]
        fig = go.Figure()
        fig.add_scatter(x=exact["date"], y=exact["price"], mode="lines+markers",
                        line=dict(color=BRONZE, width=2),
                        marker=dict(size=12, color=[PHASE_COLOURS.get(p, MOSS)
                                                    for p in exact["phase"]]),
                        customdata=exact[["event", "source"]].values,
                        hovertemplate="%{x|%d %b %Y}<br>R%{y:,.2f}<br>%{customdata[0]}"
                                      "<extra></extra>")
        fig.update_yaxes(type="log")
        show(fig, 340)
        st.caption("Sourced observations only, on a log scale. Points are coloured by phase.")
        st.markdown("<div class='note crit'><b>These are anchor points, not a price series.</b> "
                    f"{len(prices_f)} sourced observations, {int(prices_f['exact'].sum())} carrying an "
                    "exact price, and the crash-week figure is recorded as an unverified range. "
                    "They chart the collapse honestly, but they cannot train a model: there is "
                    "nothing to split into train and test, and drawing a line between them would "
                    "invent the data the models are then scored on. A daily series for 2015 to "
                    "2017 is what sub-question 4.4 needs.</div>", unsafe_allow_html=True)
        st.dataframe(prices_f[["date", "event", "price_text", "market_cap", "source", "phase"]]
                     .rename(columns={"price_text": "price (ZAR)", "market_cap": "mkt cap (Rbn)"}),
                     use_container_width=True, hide_index=True)

    st.markdown("### Chronology")
    if events_f.empty:
        st.info("No events captured.")
    else:
        rows = "".join(
            f"<tr><td class='per'>{r.period}</td><td>{r.event}</td><td>{r.category}</td>"
            f"<td>{r.indicator}</td></tr>"
            for r in events_f.itertuples())
        st.markdown("<table class='reg'><tr><th>Period</th><th>Event</th><th>Category</th>"
                    f"<th>Indicator</th></tr>{rows}</table>",
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------
# TAB 6 — Forecast (Q4.3)
# --------------------------------------------------------------------------
with tabs[5]:
    st.markdown("### Five-year counterfactual: had the scandal not occurred")
    st.caption(f"Base year FY{base['fy']} (rand millions). Scenario: {scenario}. "
               "Change the assumptions in the sidebar.")

    cards = st.columns(5)
    kpi(cards[0], "Base revenue", rm(base["revenue"]), f"FY{base['fy']} ZAR m", MOSS)
    kpi(cards[1], "FY2021 projected revenue", rm(proj.at[2021, "revenue"]), scenario, MOSS)
    kpi(cards[2], "FY2021 projected equity", rm(proj.at[2021, "total_equity"]), scenario, MOSS)
    kpi(cards[3], "Cumulative FCFF", rm(proj["fcff"].sum()), "five years", BRONZE)
    kpi(cards[4], "Base net debt", rm(base["net_debt"]), f"FY{base['fy']}", MUSTARD)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for name in an.SCENARIOS:
            p = an.project(base, name, tax_rate)
            fig.add_scatter(x=p.index, y=p["revenue"], mode="lines+markers", name=name,
                            line=dict(width=2, dash="solid" if name == scenario else "dot"),
                            marker=dict(size=7),
                            hovertemplate=name + " FY%{x}<br>%{y:,.0f}m<extra></extra>")
        fig.add_scatter(x=[base["fy"]], y=[base["revenue"]], mode="markers", name="Base year",
                        marker=dict(size=11, color=TEXT, symbol="square"))
        show(fig, 360, legend=True)
        st.caption("Projected revenue on each scenario, from the last pre-disclosure base year.")
    with c2:
        fig = go.Figure()
        fig.add_bar(x=proj.index, y=proj["fcff"], width=.6, marker_color=MOSS, name="FCFF",
                    hovertemplate="FY%{x}<br>FCFF %{y:,.0f}m<extra></extra>")
        fig.add_scatter(x=proj.index, y=proj["ebit"], mode="lines+markers", name="EBIT",
                        line=dict(color=BRONZE, width=2), marker=dict(size=8),
                        hovertemplate="FY%{x}<br>EBIT %{y:,.0f}m<extra></extra>")
        show(fig, 360, legend=True)
        st.caption("Free cash flow to the firm and operating profit under the selected scenario.")

    st.markdown("#### Projected against reported")
    if gap.empty:
        st.info("No overlapping years to compare.")
    else:
        fig = go.Figure()
        fig.add_bar(x=gap.index, y=gap["projected_revenue"], name="Projected (no scandal)",
                    marker_color=MOSS, width=.35, offset=-.35)
        fig.add_bar(x=gap.index, y=gap["actual_revenue"], name="Reported",
                    marker_color=OXBLOOD, width=.35, offset=0)
        show(fig, 320, legend=True)
        st.markdown(
            "<div class='note'><b>Reading the gap.</b> Both series are now in rand, so the "
            "comparison is on a like basis. The FY2015 base-year revenue is still flagged in the "
            "integrity register, and the reported FY2017 and FY2018 figures include the restated "
            "position — the gap therefore shows the shape of the counterfactual against the "
            "reported outcome, not a precise quantum of value destroyed.</div>",
            unsafe_allow_html=True)

    st.markdown("#### Projected statements")
    st.dataframe(proj[["revenue", "ebitda", "ebit", "da", "capex", "change_nwc", "tax",
                       "nopat", "fcff", "total_assets", "total_equity", "net_debt"]].round(0),
                 use_container_width=True)


# --------------------------------------------------------------------------
# TAB 7 — Valuation (Q4.5)
# --------------------------------------------------------------------------
with tabs[6]:
    st.markdown("### Intrinsic value and sensitivity")
    st.caption(f"Discounted cash flow on the {scenario} scenario at a WACC of {wacc_v:.2%} "
               f"and terminal growth of {term_g:.1%}.")

    cards = st.columns(5)
    kpi(cards[0], "Enterprise value", rm(val["enterprise_value"]), "ZAR m", MOSS)
    kpi(cards[1], "Equity value", rm(val["equity_value"]), "less net debt",
        MOSS if val["equity_value"] > 0 else OXBLOOD)
    kpi(cards[2], "Value per share", f"R{val['value_per_share']:,.2f}",
        f"{shares_m:,.0f}m shares", BRONZE)
    kpi(cards[3], "Terminal value share",
        pct(val["pv_terminal"] / val["enterprise_value"]) if val["enterprise_value"] else "—",
        "of enterprise value", MUSTARD)
    kpi(cards[4], "Cost of equity", pct(ke), f"beta {beta:.2f}", PLUM)

    c1, c2 = st.columns([2, 3])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=list(val["pv_by_year"].index) + ["Terminal"],
                    y=list(val["pv_by_year"].values) + [val["pv_terminal"]],
                    marker_color=[MOSS] * len(val["pv_by_year"]) + [MUSTARD], width=.6,
                    hovertemplate="%{x}<br>%{y:,.0f}m<extra></extra>")
        show(fig, 340)
        st.caption("Present value by year and the terminal value.")
    with c2:
        waccs = [wacc_v - .02, wacc_v - .01, wacc_v, wacc_v + .01, wacc_v + .02]
        growths = [max(term_g - .01, 0), term_g - .005, term_g, term_g + .005, term_g + .01]
        grid = an.sensitivity(proj, base["net_debt"], shares_m, waccs, growths)
        fig = go.Figure(go.Heatmap(
            z=grid.values, x=grid.columns, y=grid.index, colorscale=[[0, OXBLOOD], [.5, SURFACE], [1, MOSS]],
            hovertemplate="WACC %{y}<br>growth %{x}<br>R%{z:,.2f}<extra></extra>",
            colorbar=dict(tickfont=dict(color=MUTED, size=10))))
        fig.update_layout(xaxis_title="Terminal growth", yaxis_title="WACC")
        show(fig, 340)
        st.caption("Value per share across the two assumptions the result is most sensitive to.")

    st.markdown("#### Cross-check against other methods")
    rel = an.relative_value(proj, base, peer_pe, peer_ev, shares_m, base["net_debt"], tax_rate)
    ddm_v = an.ddm(dps, term_g, ke)
    rows = [dict(method="Discounted cash flow", equity_value=val["equity_value"],
                 value_per_share=val["value_per_share"])]
    if not rel.empty:
        rows += rel.to_dict("records")
    if ddm_v:
        rows.append(dict(method="Dividend discount model", equity_value=ddm_v * shares_m,
                         value_per_share=ddm_v))
    st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)

    if not ddm_v:
        st.markdown("<div class='note'><b>Dividend discount model not run.</b> No dividend per "
                    "share has been captured. Enter the base-year dividend in the sidebar, or "
                    "capture the dividend history, and the method appears above.</div>",
                    unsafe_allow_html=True)
    st.markdown("<div class='note'><b>Peer multiples are placeholders</b> until peer data is "
                "captured. The multiples in the sidebar are inputs, not sourced comparables, and "
                "must be replaced before anything here is quoted.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# TAB 8 — Predictive analytics (Q4.4)
# --------------------------------------------------------------------------
with tabs[7]:
    res = run_ml(ratios, "data")
    st.markdown("### Two models: Linear Regression and Decision Tree")

    if res["mode"] == "price":
        st.caption(f"Target: {res['target_label']}. Trained on {res['train_rows']:,} sessions "
                   f"before {pd.Timestamp(res['split']).date()}, tested on {res['test_rows']:,} "
                   "sessions from that date forward.")
        st.markdown("<div class='note'><b>Why the split sits where it does.</b> Training across "
                    "the collapse would let the models learn the very event they are meant to be "
                    "tested against. Training before it and testing into it asks the question the "
                    "brief actually poses.</div>", unsafe_allow_html=True)
    cards = st.columns(4)
    lin_m, tree_m = res["metrics"]["Linear Regression"]["test"], res["metrics"]["Decision Tree"]["test"]
    kpi(cards[0], "Linear Regression R²", f"{lin_m['R2']:.3f}", f"RMSE {lin_m['RMSE']:,.3f}", MOSS)
    kpi(cards[1], "Linear Regression MAE", f"{lin_m['MAE']:,.3f}", f"MAPE {lin_m['MAPE']:.1f}%", MOSS)
    kpi(cards[2], "Decision Tree R²", f"{tree_m['R2']:.3f}", f"RMSE {tree_m['RMSE']:,.3f}", BRONZE)
    kpi(cards[3], "Decision Tree MAE", f"{tree_m['MAE']:,.3f}", f"MAPE {tree_m['MAPE']:.1f}%", BRONZE)

    better = "Linear Regression" if lin_m["R2"] >= tree_m["R2"] else "Decision Tree"
    st.markdown(f"<div class='note'><b>On these measures {better} performs better.</b> "
                "R squared is the share of variation explained, RMSE penalises large errors more "
                "heavily than MAE, and MAPE expresses average error as a percentage. Both train "
                "and test figures are shown below so that overfitting is visible rather than "
                "hidden.</div>", unsafe_allow_html=True)

    mrows = []
    for name in ["Linear Regression", "Decision Tree"]:
        for split in ["train", "test"]:
            d = res["metrics"][name][split]
            mrows.append(dict(Model=name, Split=split.title(), **{
                "R²": round(d["R2"], 4), "RMSE": round(d["RMSE"], 4),
                "MAE": round(d["MAE"], 4), "MAPE %": round(d["MAPE"], 2), "n": d["n_test"]}))
    st.dataframe(pd.DataFrame(mrows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Predicted against actual")
        fig = go.Figure()
        n = min(len(res["actual"]), 400)
        x = res.get("test_index", np.arange(len(res["actual"])))[:n]
        fig.add_scatter(x=x, y=res["actual"][:n], mode="lines", name="Actual",
                        line=dict(color=TEXT, width=2))
        fig.add_scatter(x=x, y=res["predictions"]["Linear Regression"][:n], mode="lines",
                        name="Linear Regression", line=dict(color=MOSS, width=1.6))
        fig.add_scatter(x=x, y=res["predictions"]["Decision Tree"][:n], mode="lines",
                        name="Decision Tree", line=dict(color=BRONZE, width=1.6, dash="dot"))
        show(fig, 340, legend=True)
    with c2:
        st.markdown("#### What each model relies on")
        imp = res["importance"].head(8)
        fig = go.Figure()
        fig.add_bar(x=imp["importance"], y=imp["feature"], orientation="h", marker_color=BRONZE,
                    hovertemplate="%{y}<br>%{x:.3f}<extra></extra>")
        fig.update_yaxes(autorange="reversed")
        show(fig, 340)
        st.caption("Decision Tree feature importance. Linear Regression coefficients are below.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Linear Regression coefficients**")
        st.dataframe(res["coefficients"].round(4), use_container_width=True, hide_index=True)
        st.caption("Sign shows direction, magnitude shows sensitivity per unit of the feature.")
    with c2:
        st.markdown("**Decision Tree — the fitted rules**")
        st.code(res["tree_rules"][:2200], language="text")
        st.caption("The tree is readable, which is why it is worth showing beside the regression: "
                   "the thresholds it selects can be checked against the ratio panel.")

    if res["mode"] == "distress" and not res["company_scores"].empty:
        st.markdown("#### Steinhoff scored on its own ratios")
        piv = res["company_scores"].pivot(index="fy", columns="model", values="distress_index")
        fig = go.Figure()
        for name, colour in [("Linear Regression", MOSS), ("Decision Tree", BRONZE)]:
            if name in piv.columns:
                fig.add_scatter(x=piv.index, y=piv[name], mode="lines+markers", name=name,
                                line=dict(color=colour, width=2), marker=dict(size=9),
                                hovertemplate=name + " FY%{x}<br>%{y:.3f}<extra></extra>")
        phase_bands(fig)
        show(fig, 320, legend=True)
        st.caption("Distress index, 0 sound to 1 severe. Both models place the company in "
                   "distress from FY2015, two years before disclosure.")

    st.markdown("**Feature definitions**")
    st.dataframe(pd.DataFrame([{"feature": k, "definition": v}
                               for k, v in res["feature_defs"].items()]),
                 use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# TAB 9 — Data & integrity
# --------------------------------------------------------------------------
with tabs[8]:
    st.markdown("### Integrity register")

    cards = st.columns(3)
    kpi(cards[0], "Critical", str(int((checks_f.severity == "Critical").sum())),
        "balance sheet does not balance", OXBLOOD)
    kpi(cards[1], "High", str(int((checks_f.severity == "High").sum())),
        "capture and comparability", MUSTARD)
    kpi(cards[2], "Blocking", str(int((checks_f.severity == "Blocking").sum())),
        "data not captured at all", PLUM)

    show_c = checks_f.copy()
    show_c["fy"] = show_c["fy"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    st.dataframe(show_c.rename(columns={"fy": "Year", "severity": "Severity",
                                        "check": "Finding", "detail": "Detail"}),
                 use_container_width=True, hide_index=True)

    st.markdown("### Recomputed ratios")
    st.caption("Recomputed from the captured statement lines, not read from the workbook's own "
               "ratio sheet, so each figure traces to a line item.")
    st.dataframe(ratios_f.drop(columns=["fy"]).round(4), use_container_width=True)

    st.markdown("### Captured statement lines")
    st.dataframe(statements_f, use_container_width=True, hide_index=True)

    st.markdown("### Basis of preparation")
    st.markdown(f"""
<div class='note'>
Profit measure: profit for the year, applied to every ratio using profit.<br>
Total assets: non-current assets plus current assets including amounts held for sale.<br>
Current assets and liabilities: including amounts held for sale.<br>
Debt: interest-bearing borrowings, current plus non-current.<br>
Currency: every year restated in rand in this version, so no conversion is applied.<br>
Forecast base: FY{base['fy']}. Scenario assumptions and cost of capital are sidebar inputs, not findings.
</div>""", unsafe_allow_html=True)
