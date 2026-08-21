# Steinhoff International — Financial Intelligence Dashboard

**Built against `Steinhoff Data-Version 2 2026.xlsx`.**

Covers Question 4 in full: 4.1 analytics, 4.2 business intelligence, 4.3 forecasting,
4.4 machine learning, 4.5 valuation.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The capture workbook must sit beside `app.py`.

## Files

| File | Purpose |
|---|---|
| `etl.py` | Parses the workbook, recomputes ratios, runs integrity checks, builds the scorecard |
| `analytics.py` | Q4.3 counterfactual forecast, Q4.5 DCF, sensitivity, DDM, relative multiples |
| `ml.py` | Q4.4 Linear Regression and Decision Tree, both modes |
| `app.py` | The dashboard — nine tabs |
| `data/` | Optional inputs; drop `share_price.csv` here to switch the ML tab to share price |

## The nine tabs

Overview · Profitability · Financial position · Earnings quality · Red flags &
chronology · Forecast · Valuation · Predictive analytics · Data & integrity.

## Machine learning (4.4)

Two models, as specified: **Linear Regression** and **Decision Tree**. Both are
evaluated on the same measures — R², RMSE, MAE and MAPE — with train and test figures
shown side by side so overfitting is visible rather than hidden.

The module runs in one of two modes:

**Price mode** activates when `data/share_price.csv` exists with `date` and `close`
columns. Target is the closing price; features are lagged prices, moving averages, one-day
return, twenty-day volatility and relative volume. The split trains before October 2017 and
tests into the collapse — training across the collapse would let the models learn the very
event they are meant to be tested against.

**Distress mode** is the fallback while the price series is still being sourced. Both models
are fitted to a simulated population of company years to predict a distress index from ratio
features, and Steinhoff's own years are scored against them. The dashboard labels this
plainly as a substitute that does not answer 4.4 as written. Adding the price file switches
modes with no code change.

One result worth putting in the report: the Decision Tree performs poorly on price data
outside its training range, because a tree cannot extrapolate beyond the values it has seen —
it predicts flat where the linear model tracks the fall. That contrast is a real finding
about the two model families, not a bug.

## Forecast and valuation (4.3, 4.5)

The forecast runs from the last pre-disclosure base year on three scenarios
(Conservative, Base, Expansionary) and produces revenue, EBITDA, EBIT, tax, NOPAT, FCFF
and a simplified balance sheet roll-forward. The counterfactual is compared against what
was actually reported.

Valuation is a five-year DCF with terminal value, cross-checked against P/E and EV/EBITDA
multiples and, where a dividend is supplied, a dividend discount model. A sensitivity
heatmap varies WACC against terminal growth.

**Every assumption is a sidebar parameter** — tax rate, risk-free rate, equity risk premium,
beta, cost of debt, equity weight, terminal growth, shares in issue, peer multiples, dividend
per share and the rand/euro rate. None is hard-coded, so an assessor can vary them and watch
the valuation move. The dashboard states repeatedly that these are inputs, not findings.

## Design decisions to record in the report

**Ratios are recomputed, not read.** The workbook's ratio sheet is not used as an input.
Every ratio is derived in `etl.py` from the captured statement lines, so each figure traces
back to a line item.

**Bases are declared and applied consistently.** Profit for the year throughout; total assets
as non-current plus current including held-for-sale; debt as interest-bearing borrowings,
current plus non-current.

**Findings are disclosed, not corrected.** The integrity checks run on every load. No figure
is adjusted, no gap interpolated. When the corrected workbook replaces this one, everything
updates by itself.

**Single currency throughout.** Version 2 restates every year in rand, so the currency break that
affected the earlier workbook is gone and absolute values are comparable across the full period.
No conversion assumption is applied anywhere.

**Captured analytics are shown, not replaced.** The Altman Z'' panel and the partial Beneish
indices are read from the workbook's `Distress_Models` sheet as captured and displayed alongside
the recomputed ratios. The Z'' series crosses into the distress zone in FY2015, two years before
disclosure — the clearest single answer to whether warning signs existed.

## Known limitations, stated in the dashboard itself

- The balance sheet does not balance in FY2012–FY2016; the workbook discloses this on its own
  balance check row and the dashboard repeats it rather than correcting it
- Cash flow now runs FY2013–FY2018, so accruals and cash conversion are computable across the
  full period
- Share price exists only as six sourced anchor points, not a series. Enough to chart the collapse
  for 4.2; not enough to train a model for 4.4. A daily 2015–2017 series is what is still needed
- Dividends and peer multiples not captured; the sidebar inputs are placeholders
