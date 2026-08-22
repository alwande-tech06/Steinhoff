# Steinhoff International — Financial Intelligence Dashboard

## What this is

Steinhoff was once one of the largest retail groups in the world — the owner of
Pep, Ackermans, Mattress Firm, Poundland and many other household names. In
December 2017 it disclosed "accounting irregularities," and the share price lost
roughly 85–98% of its value in a single week.

This is an interactive report — think of it as a smart version of a spreadsheet
you can click through — built from Steinhoff's own published financial
statements (2012–2018). It turns those numbers into charts, warning-sign
scorecards, and "what if" projections, so the story of the collapse can be
explored rather than just read about.

Nothing in it is invented. Every chart and number traces back to a line in the
company's own accounts, or is clearly labelled as an assumption you can change
yourself.

## How to open it

Someone with the files needs to run two commands once:

```
pip install -r requirements.txt
streamlit run app.py
```

That starts the dashboard and opens it in a web browser. The data file
(`Steinhoff Data-Version 2 2026.xlsx`) needs to be sitting in the same folder
as the program — it's already there if you have this whole project folder.

## What you'll see — the nine tabs

**Overview** — the big picture. How big the company was, and roughly when
things turned. A row of dials shows a handful of key health measures at a
glance, coloured the same way a car dashboard warning light would be: green is
fine, amber is a concern, red is a problem.

**Profitability** — was the company actually making money? Tracks profit
margins and returns year by year.

**Financial position** — could Steinhoff pay its bills, and how much did it
owe? This is where the company's liquidity (cash and short-term assets versus
short-term debts) and overall debt load are tracked.

**Earnings quality** — not every "asset" on a balance sheet is cash in the
bank. A lot of Steinhoff's reported value sat in *goodwill* — essentially the
premium paid to acquire other companies, which only has value if those
acquisitions turn out to be worth what was paid. This tab tracks how much of
the company's assets were built on that kind of soft, judgement-based value,
plus a couple of standard fraud-detection formulas (the Altman Z-score and
Beneish M-score) run on the same numbers.

**Red flags & chronology** — a scorecard of warning signs, year by year, and a
timeline of what actually happened (the acquisition spree, the CEO's abrupt
resignation, the share-price crash, the investigation). Read down the
scorecard and the warnings start appearing years before the public disclosure.

**Forecast** — a "what if the scandal had never happened?" projection: taking
the last clean pre-scandal year and projecting five years forward under three
scenarios, then comparing that imagined future against what Steinhoff actually
reported.

**Valuation** — what might the company actually be worth? Several standard
methods are run side by side (discounted cash flow, and comparisons against
similar retail companies) so you can see whether they broadly agree or not.

**Predictive analytics** — two machine-learning models (a straightforward one
and a more flexible tree-based one) trying to spot trouble before it happens,
evaluated openly on how well each one actually performed rather than just
picking whichever looks best.

**Data & integrity** — the fine print. Every data-quality issue found while
building this — numbers that don't add up, figures that had to be inferred,
things that are still missing — is logged here rather than quietly fixed.
Nothing is corrected behind the scenes; if a figure looks wrong, it's flagged,
not changed.

## The controls on the left

**Financial years** — a slider to zoom the historical tabs into a shorter
window (e.g. just the years leading up to the scandal). The forward-looking
tabs (Forecast, Valuation, Predictive analytics) are unaffected, since those
already start from a fixed point and look ahead.

**Scandal timeline** — two sliders marking when you consider the "before,"
"during" and "after" periods to begin and end. Moving them recolours every
chart's shaded background instantly, so you can test whether the story still
holds under a different reading of when things turned.

**Assumptions** — every number that goes into the forecast and valuation (tax
rate, growth rates, cost of capital, share count, comparable-company
multiples, dividend, exchange rate) is a slider or input box here, not a
number buried in the code. Nothing is hard-coded, so anyone can test how
sensitive the conclusions are to the assumptions behind them.

## The honest bits — what we're not fully sure about

Some of the source documents didn't state their units clearly (was a number in
euros or euro-cents? millions or billions?). Wherever a number had to be
inferred rather than read directly, it's marked **FLAGGED** and the reasoning
is written out — usually "this only makes sense as X, because otherwise the
result would be impossible" (e.g. a dividend bigger than the share price).

A few figures for the peer companies used in the valuation cross-check are
outright wrong in the source file — one company's earnings figure produces a
result that isn't credible for a retailer its size — and that's called out
explicitly rather than smoothed over.

Steinhoff's own historical share price only exists as a handful of anchor
points, not a full daily record. That's enough to chart the crash, but not
enough to properly train a predictive model on — so the machine-learning tab
runs a clearly-labelled substitute analysis until a real daily price history
is available.

The company's own balance sheet doesn't balance in several of the years
before the scandal broke (assets don't equal what the accounts say they
should) — this is disclosed by Steinhoff's own workbook, not something this
dashboard fixes. That mismatch is itself one of the earliest warning signs.

## The bottom line

Read down the warning-sign scorecard, or the Altman Z-score panel: distress
signals were visible as early as **2015** — two full years before the
company told the market anything was wrong in December 2017.
