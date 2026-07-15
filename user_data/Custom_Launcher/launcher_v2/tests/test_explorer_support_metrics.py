from user_data.Custom_Launcher.explorer.explorer_support import metric_summary


def test_metric_summary_derives_percent_from_canonical_profit_ratio() -> None:
    summary = metric_summary({"profit_total": 0.00844634308})

    assert summary["profit_total"] == 0.00844634308
    assert summary["profit_total_pct"] == 0.844634308


def test_metric_summary_preserves_explicit_profit_percent() -> None:
    summary = metric_summary({"profit_total": 0.1234, "profit_total_pct": 12.35})

    assert summary["profit_total_pct"] == 12.35
