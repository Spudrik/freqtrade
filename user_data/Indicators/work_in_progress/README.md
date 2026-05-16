# Work In Progress Indicators

This folder holds indicator-like modules that are not part of the active
strategy-facing indicator set.

The external context modules parked here need much deeper historical data,
data-quality validation, runtime review, and forward-return testing before they
should be imported by strategy files or exposed to hyperopt:

- `external_orderbook_context_features.py`
- `external_global_context_features.py`
- `external_news_web_sentiment_features.py`

Do not wire these into strategies unless the user explicitly reopens
external-context research.
