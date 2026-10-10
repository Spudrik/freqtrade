"""Focused mechanics and safety checks for the four new aggressive PAPER families."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from freqtrade.enums import RunMode
from freqtrade.enums import TradingMode
from freqtrade.persistence import Order, Trade
from user_data.strategies import paper_aggressive_context as context
from user_data.strategies.paper_aggressive_alt import (
    PaperAggressiveAuction, PaperAggressiveReclaim, PaperAggressiveRotation,
    PaperAggressiveVacuum, _PaperAggressiveAlt, rank_aggressive_candidates,
)

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
SOURCE = "https://example.org/reviewed-market-source"
PAIRS = tuple(sorted(context.AGGRESSIVE_PAIRS))
CLASSES = {
    "vacuum": PaperAggressiveVacuum,
    "reclaim": PaperAggressiveReclaim,
    "auction": PaperAggressiveAuction,
    "rotation": PaperAggressiveRotation,
}
OBSERVED_ROUTE_CASES = (
    ("vacuum", "profile_acceptance", "long", "vah_1h"),
    ("vacuum", "profile_acceptance", "short", "val_1h"),
    ("auction", "accepted_value_escape", "long", "vah_4h"),
    ("auction", "accepted_value_escape", "short", "val_4h"),
    ("auction", "value_edge_rotation", "long", "val_4h"),
    ("auction", "value_edge_rotation", "short", "vah_4h"),
)


def config(account="aggressive_vacuum"):
    from user_data.Custom_Launcher.research.paper_trial_control import _configs
    base, overlay = _configs(account)
    merged = deepcopy(base)
    for key, value in overlay.items():
        if key == "exchange":
            merged.setdefault(key, {}).update(value)
        else:
            merged[key] = deepcopy(value)
    merged["runmode"] = RunMode.DRY_RUN
    return merged


def signal_frame(family, *, side="long", primary=False):
    dates = pd.to_datetime([NOW - timedelta(minutes=20), NOW - timedelta(minutes=5)], utc=True)
    frame = pd.DataFrame({"date": dates})
    base = {
        "open_15m": [100., 100.], "high_15m": [101., 101.], "low_15m": [99., 99.],
        "close_15m": [100., 100.], "ag_atr_15m": [1., 1.], "ag_pressure_15m": [0., 0.],
        "ag_pressure_change_15m": [0., 0.], "ag_relative_volume_15m": [1., 1.],
        "ag_range_expansion_15m": [0.7, 0.7], "ag_atr5": [0.5, 0.5],
        "ag_day_range_high_1d": [110., 110.], "ag_day_range_low_1d": [90., 90.],
        "ag_vah_1h": [101., 101.], "ag_val_1h": [99., 99.],
        "ag_thin_above_1h": [0.2, 0.2], "ag_thin_below_1h": [0.2, 0.2],
        "ag_prior_high_15m": [110., 110.], "ag_prior_low_15m": [90., 90.],
        "ag_vah_4h": [105., 105.], "ag_val_4h": [95., 95.], "ag_poc_4h": [100., 100.],
        "ag_poc_migration_4h": [0., 0.], "ag_cohort_return": [0., 0.],
        "ag_cohort_excess": [0.001, 0.001], "ag_cohort_rank": [0.8, 0.8],
        "ag_cohort_dispersion": [0.002, 0.002],
    }
    for name, values in base.items():
        frame[name] = values

    if family == "vacuum":
        if primary:
            frame.loc[1, ["close_15m", "ag_vah_1h", "ag_val_1h", "ag_pressure_15m",
                          "ag_relative_volume_15m", "ag_thin_above_1h", "ag_thin_below_1h"]] = (
                [103., 100., 98., .5, 1.2, .8, .1] if side == "long"
                else [95., 102., 98., -.5, 1.2, .1, .8])
        else:
            frame.loc[1, ["close_15m", "ag_vah_1h", "ag_val_1h", "ag_thin_above_1h",
                          "ag_thin_below_1h"]] = ([99.5, 101., 98., .1, .9] if side == "short"
                                                    else [99.5, 101., 98., .9, .1])
    elif family == "reclaim":
        if primary:
            frame.loc[1, ["open_15m", "high_15m", "low_15m", "close_15m",
                          "ag_prior_high_15m", "ag_prior_low_15m", "ag_pressure_change_15m"]] = (
                [100., 101., 99.5, 101., 110., 100., .5] if side == "long"
                else [101., 100.5, 99., 99., 100., 90., -.5])
        else:
            frame.loc[1, ["close_15m", "ag_prior_high_15m", "ag_prior_low_15m"]] = (
                [95., 110., 90.] if side == "long" else [105., 110., 90.])
    elif family == "auction":
        if primary:
            frame.loc[0, "close_15m"] = 100.5 if side == "long" else 94.5
            frame.loc[1, ["open_15m", "high_15m", "low_15m", "close_15m", "ag_vah_4h",
                          "ag_val_4h", "ag_poc_4h", "ag_range_expansion_15m",
                          "ag_relative_volume_15m"]] = (
                [100.5, 102., 100., 101.5, 100., 95., 98., 1.2, 1.2] if side == "long"
                else [94.5, 95., 98., 98., 105., 100., 102., 1.2, 1.2])
        else:
            frame.loc[1, ["open_15m", "high_15m", "low_15m", "close_15m", "ag_vah_4h",
                          "ag_val_4h", "ag_poc_4h", "ag_poc_migration_4h"]] = (
                [100., 101., 99., 100., 105., 95., 103., 0.] if side == "long"
                else [104.9, 104.8, 104., 104.8, 105., 95., 97., 0.])
    elif family == "rotation":
        value = (.004 if primary else .0001) if side == "long" else (-.004 if primary else -.0001)
        frame.loc[:, "ag_cohort_excess"] = value
        frame.loc[:, "ag_cohort_rank"] = (.95 if primary else .7) if side == "long" else (.05 if primary else .3)
        frame.loc[:, "ag_cohort_return"] = value
        frame.loc[:, "ag_cohort_dispersion"] = .004 if primary else .002
    else:
        raise AssertionError(family)
    return frame


def decision(strategy, frame):
    result = strategy.populate_entry_trend(frame.copy(), {"pair": PAIRS[0]})
    row = result.iloc[-1]
    return row, result


@pytest.mark.parametrize("close,low,high,forbidden", [
    (89.,90.,110.,"long"),(111.,90.,110.,"short"),
    (90.,90.,110.,"long"),(110.,90.,110.,"short"),
])
def test_reclaim_never_signals_an_already_invalidated_boundary(close,low,high,forbidden):
    frame=signal_frame("reclaim")
    frame.loc[1,["close_15m","ag_prior_low_15m","ag_prior_high_15m"]]=[close,low,high]
    row,_=decision(PaperAggressiveReclaim(config("aggressive_reclaim")),frame)
    assert row[f"enter_{forbidden}"]==0
    assert row["ag_signal"]  # The alternative best guess remains available.


def test_reclaim_refuses_both_sides_when_no_boundary_survives():
    frame=signal_frame("reclaim")
    frame.loc[1,["close_15m","ag_prior_low_15m","ag_prior_high_15m"]]=[100.,100.,100.]
    row,_=decision(PaperAggressiveReclaim(config("aggressive_reclaim")),frame)
    assert not row["ag_signal"] and row["enter_long"]==row["enter_short"]==0


@pytest.mark.parametrize("side,open_,high,low,close,change,expected_anchor", [
    ("short", 105., 109.75, 89.5, 109.5, .22, 110.),
    ("long", 95., 110.5, 90.25, 90.5, -.22, 90.),
])
def test_reclaim_opposite_side_primary_does_not_change_exploratory_route_geometry(
        side, open_, high, low, close, change, expected_anchor):
    frame = signal_frame("reclaim")
    frame.loc[1, ["open_15m", "high_15m", "low_15m", "close_15m", "ag_atr_15m",
                  "ag_pressure_change_15m", "ag_prior_high_15m", "ag_prior_low_15m",
                  "ag_day_range_high_1d", "ag_day_range_low_1d"]] = [
                      open_, high, low, close, 2., change, 110., 90., 110., 90.]
    strategy = PaperAggressiveReclaim(config("aggressive_reclaim"))
    row, _ = decision(strategy, frame)

    assert row["ag_decision_side"] == side
    assert row["ag_mode"] == "exploratory"
    assert row["ag_route"] == "best_guess"
    assert "best-guess" in row["ag_reason"]
    _, _, anchor, _ = strategy._geometry(row, side, close)
    assert anchor == expected_anchor


@pytest.mark.parametrize("family,side", [(family, side) for family in CLASSES for side in ("long", "short")])
def test_each_family_has_primary_two_sided_route(family, side):
    row, _ = decision(CLASSES[family](config(f"aggressive_{family}")), signal_frame(family, side=side, primary=True))
    assert int(row["enter_long"] if side == "long" else row["enter_short"]) == 1
    assert row["ag_mode"] == "primary"
    assert row["ag_decision_side"] == side
    if family == "auction":
        assert row["ag_route"] == "accepted_value_escape"
        assert "escape" in row["ag_reason"]
    elif family == "vacuum":
        assert "historical traded-volume estimate" in row["ag_reason"]
        assert "candle-derived pressure" in row["ag_reason"]


@pytest.mark.parametrize("family,side", [(family, side) for family in CLASSES for side in ("long", "short")])
def test_family_best_guess_stays_active_without_primary_setup(family, side):
    row, _ = decision(CLASSES[family](config(f"aggressive_{family}")), signal_frame(family, side=side, primary=False))
    assert int(row["enter_long"] + row["enter_short"]) == 1
    assert row["ag_mode"] == "exploratory"
    assert row["ag_decision_side"] == side
    assert row["ag_score"] > 0
    if family in {"vacuum", "auction"}:
        assert row["ag_route"] == "best_guess"
    if family == "vacuum":
        assert "candle-derived pressure estimate" in row["ag_reason"]


@pytest.mark.parametrize("family,side,frozen,current,close,pressure", [
    ("vacuum", "long", 101., 99., 100., -.3),
    ("vacuum", "short", 99., 101., 100., .3),
    ("auction", "long", 95., 93., 94., 0.),
    ("auction", "short", 105., 107., 106., 0.),
])
def test_weak_vacuum_and_auction_guesses_do_not_claim_observed_failure(
        family, side, frozen, current, close, pressure):
    strategy = CLASSES[family](config(f"aggressive_{family}"))
    feature = ("vah_1h" if side == "long" else "val_1h") if family == "vacuum" else (
        "val_4h" if side == "long" else "vah_4h")
    plan = {"family": family, "route": "best_guess", "structural_anchor": frozen,
            "features": {feature: frozen}}
    row = {"close_15m": close, "ag_pressure_15m": pressure,
           "ag_vah_1h": current, "ag_val_1h": current,
           "ag_vah_4h": current, "ag_val_4h": current}
    assert strategy._failure_exit_reason(row, plan, side) is None


@pytest.mark.parametrize("family,route,side,feature,frozen,current,close,pressure,expected", [
    ("vacuum", "profile_acceptance", "long", "vah_1h", 101., 99., 100., -.3,
     "vacuum_profile_acceptance_failed"),
    ("vacuum", "profile_acceptance", "short", "val_1h", 99., 101., 100., .3,
     "vacuum_profile_acceptance_failed"),
    ("auction", "accepted_value_escape", "long", "vah_4h", 101., 99., 100., 0.,
     "auction_accepted_escape_failed"),
    ("auction", "accepted_value_escape", "short", "val_4h", 99., 101., 100., 0.,
     "auction_accepted_escape_failed"),
    ("auction", "value_edge_rotation", "long", "val_4h", 99., 97., 98., 0.,
     "auction_value_edge_rotation_failed"),
    ("auction", "value_edge_rotation", "short", "vah_4h", 101., 103., 102., 0.,
     "auction_value_edge_rotation_failed"),
])
def test_observed_profile_failures_use_frozen_entry_features(
        family, route, side, feature, frozen, current, close, pressure, expected):
    strategy = CLASSES[family](config(f"aggressive_{family}"))
    plan = {"family": family, "route": route, "structural_anchor": frozen,
            "features": {feature: frozen}}
    row = {"close_15m": close, "ag_pressure_15m": pressure,
           "ag_vah_1h": current, "ag_val_1h": current,
           "ag_vah_4h": current, "ag_val_4h": current}
    assert strategy._failure_exit_reason(row, plan, side) == expected


@pytest.mark.parametrize("side,open_,high,low,close,route_reason", [
    ("long", 95.4, 96., 95., 95.5, "observed lower"),
    ("short", 104.8, 105., 104., 104.5, "observed upper"),
])
def test_auction_edge_rejection_route_and_reason_match_winning_side(
        side, open_, high, low, close, route_reason):
    frame = signal_frame("auction", side=side, primary=False)
    frame.loc[1, ["open_15m", "high_15m", "low_15m", "close_15m", "ag_vah_4h",
                  "ag_val_4h", "ag_poc_4h", "ag_range_expansion_15m",
                  "ag_relative_volume_15m"]] = [open_, high, low, close, 105., 95., 100., .7, 1.]
    row, _ = decision(PaperAggressiveAuction(config("aggressive_auction")), frame)
    assert row["ag_decision_side"] == side
    assert row["ag_route"] == "value_edge_rotation"
    assert route_reason in row["ag_reason"]


def test_auction_route_follows_winning_side_when_opposite_escape_fires():
    frame = signal_frame("auction", side="short", primary=True)
    frame.loc[0, "close_15m"] = 94.8
    frame.loc[1, ["open_15m", "high_15m", "low_15m", "close_15m", "ag_vah_4h",
                  "ag_val_4h", "ag_poc_4h", "ag_range_expansion_15m",
                  "ag_relative_volume_15m", "ag_poc_migration_4h"]] = [
                      90., 91., 89., 90., 105., 95., 100., 1.2, 1.2, .02]
    row, _ = decision(PaperAggressiveAuction(config("aggressive_auction")), frame)
    assert row["ag_decision_side"] == "long"
    assert row["ag_mode"] == "exploratory"
    assert row["ag_route"] == "best_guess"
    assert "weak anticipated long" in row["ag_reason"]


@pytest.mark.parametrize("excess,rank,own_return,close,side", [
    (.004, .95, .001, 90., "long"),   # positive excess wins despite daily low / clipping tie
    (-.004, .05, -.001, 110., "short"), # negative excess wins despite daily high
    (0., .5, .001, 90., "long"),       # own return breaks a neutral cohort tie
    (0., .5, -.001, 110., "short"),
])
def test_rotation_direction_cannot_be_reversed_by_daily_position(
        excess, rank, own_return, close, side):
    frame = signal_frame("rotation", side="long", primary=False)
    frame.loc[1, ["close_15m", "ag_cohort_excess", "ag_cohort_rank",
                  "ag_cohort_return", "ag_cohort_dispersion"]] = [
                      close, excess, rank, own_return, .004]
    row, _ = decision(PaperAggressiveRotation(config("aggressive_rotation")), frame)
    assert row["ag_decision_side"] == side


@pytest.mark.parametrize("family,mutate", [
    ("vacuum", lambda frame: frame.assign(ag_thin_above_1h=.1, ag_thin_below_1h=.9)),
    ("reclaim", lambda frame: frame.assign(ag_pressure_change_15m=-.8)),
    ("auction", lambda frame: frame.assign(ag_poc_migration_4h=-.02)),
    ("rotation", lambda frame: frame.assign(ag_cohort_excess=-.0001, ag_cohort_rank=.05,
                                              ag_cohort_return=-.0001)),
])
def test_family_specific_input_changes_exploratory_decision(family, mutate):
    strategy = CLASSES[family](config(f"aggressive_{family}"))
    original, _ = decision(strategy, signal_frame(family, side="long", primary=False))
    changed, _ = decision(strategy, mutate(signal_frame(family, side="long", primary=False)))
    assert original["ag_mode"] == changed["ag_mode"] == "exploratory"
    assert original["ag_decision_side"] != changed["ag_decision_side"]


def _filled_plan(side="long"):
    direction = 1 if side == "long" else -1
    entry = 100.
    initial_stop = 97. if side == "long" else 103.
    target = 110. if side == "long" else 90.
    quantity, leverage, stake, equity = 50., 5., 1000., 10000.
    notional = quantity * entry
    risk = notional * (abs(entry-initial_stop)/entry + context.AGGRESSIVE_LIMITS["fee_allowance"])
    return {"family": "vacuum", "mode": "primary", "route": "profile_acceptance",
        "side": side, "pair": PAIRS[0], "entry_tag": f"aggressive:vacuum:primary:profile_acceptance:{side}",
        "provenance": "automatic_signal", "reason": "test mechanics; not an edge claim", "features": {},
        "reference_rate": entry, "open_rate": entry, "entry_atr": 2., "atr15": 4.,
        "structural_anchor": initial_stop, "approved_stop_price": initial_stop,
        "fill_stop_adjustment": None, "stop_price": initial_stop, "initial_stop_price": initial_stop,
        "target_price": target, "target_kind": "next_observed_4h_range", "planned_loss_usdt": risk,
        "risk_reserve_usdt": risk*1.1, "entry_equity_usdt": equity, "stake_usdt": stake,
        "quantity": quantity, "leverage": leverage, "requested_leverage": leverage,
        "contract_size": 1., "filled_at_utc": NOW.isoformat()}


def _pending_fill(side="long", *, reserve=175.):
    stop, target = (97., 110.) if side == "long" else (103., 90.)
    return {"family": "vacuum", "mode": "manual", "route": "best_guess", "pair": PAIRS[0],
        "side": side, "entry_tag": "aggressive_manual:fill-test", "provenance": "main_agent_force_entry",
        "reason": "test fill", "features": {}, "reference_rate": 100., "entry_atr": 2.,
        "atr15": 4., "structural_anchor": stop, "stop_price": stop, "target_price": target,
        "target_kind": "main_agent_approved", "risk_reserve_usdt": reserve,
        "entry_equity_usdt": 10000., "stake_usdt": 1000., "requested_leverage": 5., "leverage": 5.}


def _freqtrade_trade_order(*, trade_id, pair, entry_tag, side, rate, amount, stake, leverage):
    trade = Trade(id=trade_id, exchange="binance", pair=pair, is_open=True,
        fee_open=.001, fee_close=.001, open_rate=rate, stake_amount=stake, amount=amount,
        open_date=NOW - timedelta(minutes=30), strategy="PaperAggressiveVacuum",
        enter_tag=entry_tag, leverage=leverage, is_short=side == "short",
        trading_mode=TradingMode.FUTURES, contract_size=1.)
    order = Order(order_id=f"aggressive-entry-{trade_id}", ft_order_side=trade.entry_side,
        ft_pair=pair, ft_amount=amount, ft_price=rate)
    return trade, order


@pytest.mark.parametrize("side,valid_tight,profit_lock,loose", [
    ("long", 98., 100.5, 96.), ("short", 102., 99.5, 104.)])
def test_filled_protection_accepts_tightening_and_profit_locks_rejects_loosened(side, valid_tight, profit_lock, loose):
    base = _filled_plan(side)
    context.validate_aggressive_filled_plan(base, actual_pair=base["pair"], actual_side=side,
        actual_open_rate=base["open_rate"], actual_quantity=base["quantity"],
        actual_stake=base["stake_usdt"], actual_leverage=base["leverage"])
    for stop in (valid_tight, profit_lock):
        tightened = dict(base, stop_price=stop)
        context.validate_aggressive_filled_plan(tightened)
    with pytest.raises(ValueError, match="loosened"):
        context.validate_aggressive_filled_plan(dict(base, stop_price=loose))
    with pytest.raises(ValueError, match="target"):
        context.validate_aggressive_filled_plan(dict(base, stop_price=base["target_price"]))


def test_actual_fill_reconciles_normalized_quantity_and_caps_risk():
    pending = {"family": "vacuum", "mode": "exploratory", "route": "profile_acceptance",
        "pair": PAIRS[0], "side": "long", "entry_tag": "aggressive:vacuum:exploratory:profile_acceptance:long",
        "provenance": "automatic_signal", "reason": "test fill", "features": {"ag_pressure_15m": .3},
        "reference_rate": 100., "entry_atr": 2., "atr15": 4., "structural_anchor": 97.,
        "stop_price": 97., "target_price": 110., "target_kind": "next_observed_4h_range",
        "risk_reserve_usdt": 180., "entry_equity_usdt": 10000., "stake_usdt": 1000.,
        "requested_leverage": 5., "leverage": 5.}
    quantity = 1000. * 5. / 101.
    plan = context.build_aggressive_filled_plan(pending, pair=PAIRS[0], side="long", open_rate=101.,
        quantity=quantity, stake=1000., leverage=5., contract_size=1., filled_at=NOW)
    assert plan["open_rate"] == 101. and plan["quantity"] == pytest.approx(quantity)
    assert plan["target_price"] == 110. and plan["target_kind"] == "next_observed_4h_range"
    assert plan["approved_stop_price"] == 97.
    assert plan["fill_stop_adjustment"] == "risk_budget_tightening"
    assert plan["stop_price"] > plan["approved_stop_price"]
    assert plan["planned_loss_usdt"] <= plan["risk_reserve_usdt"]
    with pytest.raises(ValueError, match="do not reconcile"):
        context.build_aggressive_filled_plan(pending, pair=PAIRS[0], side="long", open_rate=101.,
            quantity=quantity*1.05, stake=1000., leverage=5., contract_size=1., filled_at=NOW)


@pytest.mark.parametrize("side,favorable_rate,adverse_rate,approved_stop", [
    ("long", 99., 101., 97.), ("short", 101., 99., 103.)])
def test_actual_fill_preserves_absolute_stop_or_marks_tighter_risk_adjustment(
        side, favorable_rate, adverse_rate, approved_stop):
    pending = _pending_fill(side)
    target = pending["target_price"]
    for fill_rate, should_tighten in ((favorable_rate, False), (adverse_rate, True)):
        quantity = pending["stake_usdt"] * pending["requested_leverage"] / fill_rate
        plan = context.build_aggressive_filled_plan(pending, pair=PAIRS[0], side=side,
            open_rate=fill_rate, quantity=quantity, stake=1000., leverage=5.,
            contract_size=1., filled_at=NOW)
        assert plan["approved_stop_price"] == approved_stop
        assert plan["target_price"] == target
        if should_tighten:
            assert plan["fill_stop_adjustment"] == "risk_budget_tightening"
            assert (plan["initial_stop_price"] > approved_stop if side == "long"
                    else plan["initial_stop_price"] < approved_stop)
        else:
            assert plan["fill_stop_adjustment"] is None
            assert plan["initial_stop_price"] == approved_stop
        assert plan["stop_price"] == plan["initial_stop_price"]
        context.validate_aggressive_filled_plan(plan)


def test_actual_fill_preserves_target_at_the_existing_fee_meaningful_floor():
    pending = _pending_fill("long")
    pending["target_price"] = 100. * (1. + 2.5 * context.AGGRESSIVE_LIMITS["fee_allowance"])
    plan = context.build_aggressive_filled_plan(pending, pair=PAIRS[0], side="long",
        open_rate=100., quantity=50., stake=1000., leverage=5.,
        contract_size=1., filled_at=NOW)
    assert plan["target_price"] == pending["target_price"]
    assert plan["target_kind"] == "main_agent_approved"


def control_row(account="paper_aggressive_vacuum", *, expires=None, **changes):
    row = {"schema_version": 1, "author": "main_agent", "account": account,
        "decision_id": "control-1", "observed_at_utc": NOW.isoformat(),
        "valid_until_utc": (expires or NOW + timedelta(hours=4)).isoformat(),
        "bias": 0, "side_permission": "both", "long_leverage_cap": 8.,
        "short_leverage_cap": 8., "exposure": "normal", "reason": "conditional observation",
        "sources": [SOURCE], "unavailable_inputs": []}
    row.update(changes)
    return row


def test_expiring_account_bound_controls_and_technical_only_fallback(tmp_path):
    path = tmp_path / "control.json"
    missing = context.load_aggressive_control("aggressive_vacuum", NOW, path)
    assert missing.status == "missing" and missing.technical_only
    assert missing.leverage_cap("long") == 3.
    path.write_text("{", encoding="utf-8")
    assert context.load_aggressive_control("aggressive_vacuum", NOW, path).status == "malformed"
    path.write_text(json.dumps(control_row("paper_aggressive_reclaim")), encoding="utf-8")
    assert context.load_aggressive_control("aggressive_vacuum", NOW, path).status == "malformed"
    path.write_text(json.dumps(control_row(expires=NOW - timedelta(hours=1),
        observed_at_utc=(NOW - timedelta(hours=5)).isoformat())), encoding="utf-8")
    assert context.load_aggressive_control("aggressive_vacuum", NOW, path).status == "stale"
    parsed = context.parse_aggressive_control(control_row(bias=2), NOW, "aggressive_vacuum")
    assert parsed.desired_leverage("long", "primary", .95) == 8.
    with pytest.raises(ValueError, match="different PAPER account"):
        context.parse_aggressive_control(control_row("paper_aggressive_reclaim"), NOW, "aggressive_vacuum")


def test_manual_force_plan_overrides_automatic_side_but_not_expiry_risk_or_cap(monkeypatch):
    plan = {"pair": PAIRS[0], "side": "short", "reference_rate": 100., "stake_pct": .02,
        "leverage": 3., "stop_price": 102., "take_profit_price": 95.,
        "valid_until_utc": (NOW + timedelta(minutes=5)).isoformat(),
        "review_due_at_utc": (NOW + timedelta(hours=4)).isoformat()}
    control_row_long = context.parse_aggressive_control(control_row(side_permission="long_only"), NOW)
    context.validate_aggressive_manual_plan(plan, 100., "aggressive_vacuum", equity=10000.,
        control=control_row_long, now=NOW)
    plan["leverage"] = 9.
    with pytest.raises(ValueError, match="current side-specific leverage cap"):
        context.validate_aggressive_manual_plan(plan, 100., "aggressive_vacuum",
            control=control_row_long, now=NOW)
    plan["leverage"] = 3.
    plan["valid_until_utc"] = (NOW - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="expired"):
        context.validate_aggressive_manual_plan(plan, 100., "aggressive_vacuum", now=NOW)


def test_context_bias_and_btc_eth_change_risk_size_without_direction_veto():
    strategy = PaperAggressiveVacuum(config())
    neutral = context.AggressiveControl("observed", bias=0)
    aligned = context.AggressiveControl("observed", bias=2)
    row = {"ag_btc_return45": .02, "ag_eth_return45": .01}
    long_factor = strategy._context_size_factor(neutral, "long", row)
    short_factor = strategy._context_size_factor(neutral, "short", row)
    assert long_factor > short_factor
    assert strategy._context_size_factor(aligned, "long", row) > long_factor
    technical = context.AggressiveControl("missing")
    assert strategy._context_size_factor(technical, "long", row) == pytest.approx(long_factor)
    assert technical.desired_leverage("long", "primary", .99) == 3.


def test_candidate_rank_ignores_whitelist_order_and_occupied_pairs(monkeypatch):
    assert rank_aggressive_candidates([
        {"pair": PAIRS[0], "score": .5}, {"pair": PAIRS[1], "score": .9}], 1) == [PAIRS[1]]
    strategy = PaperAggressiveVacuum(config())
    strategy.config["max_open_trades"] = 3
    occupied = PAIRS[2]
    rows = {}
    for index, pair in enumerate(PAIRS):
        analyzed = strategy.populate_entry_trend(
            signal_frame("vacuum", side="long", primary=True), {"pair": pair})
        analyzed["date"] = NOW - timedelta(minutes=5)
        analyzed["close"] = 100.
        analyzed["ag_signal"] = index < 2
        analyzed["ag_score"] = .99 if index == 0 else .75
        analyzed["ag_decision_side"] = "long"
        analyzed["ag_mode"] = "primary"
        analyzed["ag_vah_1h"] = 50. if index == 0 else 98.5
        rows[pair] = analyzed

    class CachedExchange:
        def get_max_leverage(self, pair, stake):
            return 10.
        def get_min_pair_stake_amount(self, pair, rate, stoploss, leverage):
            return 1.
        def get_max_pair_stake_amount(self, pair, rate, leverage):
            return 1500.

    class CachedWallets:
        def get_total_stake_amount(self):
            return 10000.
        def get_available_stake_amount(self):
            return 9000.
        def get_trade_stake_amount(self, pair, max_open_trades, update=True):
            return 1000.

    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (rows[pair], None),
                                  _exchange=CachedExchange())
    strategy.wallets = CachedWallets()
    strategy._open_risk_reserve = lambda equity: 0.
    monkeypatch.setattr("user_data.strategies.paper_aggressive_alt.Trade.get_open_trades",
                        lambda: [SimpleNamespace(pair=occupied), SimpleNamespace(pair="other/USDT:USDT")])
    strategy._control = lambda _now: context.AggressiveControl(
        "observed", "rank-control", 0, "both", 7., 7., "normal")
    assert strategy._top_rank_allows(PAIRS[1], NOW)
    assert not strategy._top_rank_allows(PAIRS[0], NOW)
    assert not strategy._top_rank_allows(occupied, NOW)


class DataProvider:
    def __init__(self, base):
        self.data = {}
        for index, pair in enumerate((*PAIRS, *context.LEADER_PAIRS)):
            self.data[(pair, "5m")] = base(index)
            for timeframe, rule in (("15m", "15min"), ("1h", "1h"), ("4h", "4h")):
                self.data[(pair, timeframe)] = self._resample(self.data[(pair, "5m")], rule)
            self.data[(pair, "1d")] = self._daily(index)

    @staticmethod
    def _resample(frame, rule):
        source = frame.set_index("date")
        result = source.resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        return result.reset_index()

    @staticmethod
    def _daily(index):
        dates = pd.date_range(end=pd.Timestamp(NOW).floor("1D"), periods=65, freq="1D", tz="UTC")
        base = 40. + index * 8.
        close = base + np.arange(len(dates)) * (.04 + index * .001) + np.sin(np.arange(len(dates)) / 4)
        opened = np.r_[close[0] - .2, close[:-1]]
        return pd.DataFrame({"date": dates, "open": opened, "high": np.maximum(opened, close) + .6,
            "low": np.minimum(opened, close) - .6, "close": close, "volume": 1000. + np.arange(len(dates))})

    def get_pair_dataframe(self, pair, timeframe):
        return self.data[(pair, timeframe)].copy()


def realistic_provider():
    end = pd.Timestamp(NOW) - pd.Timedelta(minutes=5)
    dates = pd.date_range(end=end, periods=2100, freq="5min", tz="UTC")

    def base(index):
        i = np.arange(len(dates), dtype=float)
        close = 40. + index * 8. + i * (.001 + index * .00003) + np.sin(i / (8. + index)) * .35
        opened = np.r_[close[0] - .03, close[:-1]]
        return pd.DataFrame({"date": dates, "open": opened,
            "high": np.maximum(opened, close) + .12, "low": np.minimum(opened, close) - .12,
            "close": close, "volume": 1000. + 300 * (1 + np.sin(i / 11 + index))})
    return DataProvider(base)


@pytest.mark.parametrize("family", CLASSES)
def test_realistic_populate_indicators_to_entries_is_aligned_and_prefix_causal(family):
    strategy = CLASSES[family](config(f"aggressive_{family}"))
    strategy.dp = realistic_provider()
    pair = PAIRS[0]
    base = strategy.dp.get_pair_dataframe(pair, "5m")
    metadata = {"pair": pair}
    full = strategy.populate_entry_trend(strategy.populate_indicators(base, metadata), metadata)
    assert {"open_15m", "high_15m", "low_15m", "close_15m"} <= set(full.columns)
    assert len(full) == len(base) and full["date"].is_unique
    assert full["ag_cohort_excess"].notna().any()
    assert full.loc[full.index[-1], "date"].minute == 55
    prefix_count = len(base) - 60
    prefix = base.iloc[:prefix_count].copy()
    prefix_result = strategy.populate_entry_trend(strategy.populate_indicators(prefix, metadata), metadata)
    columns = ["enter_long", "enter_short", "ag_score", "ag_mode", "ag_route", "ag_decision_side"]
    pd.testing.assert_frame_equal(full.loc[prefix.index, columns], prefix_result.loc[:, columns], check_dtype=False)


def test_actual_private_overlays_load_and_bot_start_reads_exact_local_identity():
    from freqtrade.configuration.load_config import load_from_files
    from freqtrade.resolvers import StrategyResolver
    root = Path(__file__).resolve().parents[3]
    for key, strategy_class in ((key, cls) for key, cls in CLASSES.items()):
        account = f"aggressive_{key}"
        overlay = root / context.AGGRESSIVE_ACCOUNTS[account]["config"]
        assert overlay.is_file()
        merged = load_from_files([str(root / "user_data/configs/integrated_paper_20260926_base.json"), str(overlay)])
        assert merged["bot_name"] == context.AGGRESSIVE_ACCOUNTS[account]["bot_name"]
        assert merged["strategy"] == strategy_class.__name__
        assert set(merged["exchange"]["pair_whitelist"]) == context.AGGRESSIVE_PAIRS
        assert not any(merged["exchange"].get(name) for name in ("key", "secret", "password", "privateKey", "private_key"))
        merged.update(runmode=RunMode.DRY_RUN, strategy_path=str(root / "user_data/strategies"),
                      user_data_dir=root / "user_data")
        strategy = StrategyResolver.load_strategy(merged)
        assert strategy.__class__.__name__ == strategy_class.__name__
        strategy.bot_start()


def test_journalled_main_force_entry_passes_real_strategy_callbacks_without_auto_signal(tmp_path, monkeypatch):
    from user_data.Custom_Launcher.research import paper_trial_control as controller
    from user_data.strategies import paper_news_manual

    control = context.AggressiveControl("missing")
    approval = {"decision_id": "force-no-signal", "account": "paper_aggressive_vacuum",
        "action": "enter", "status": "proposed", "reason": "explicit reviewed manual setup",
        "source": SOURCE, "plan": {"pair": PAIRS[0], "side": "short", "reference_rate": 100.,
        "stake_pct": .02, "leverage": 3., "stop_price": 102., "take_profit_price": 95.,
        "valid_until_utc": (NOW + timedelta(minutes=5)).isoformat(),
        "review_due_at_utc": (NOW + timedelta(hours=4)).isoformat()}}
    rows = []
    routes = []
    from user_data.Custom_Launcher.research import paper_trial_runtime as runtime
    record = {"process_recovery": {
        "allowed_accounts": sorted(runtime.REVIEWED_23_ACCOUNT_KEYS),
        "account_lifecycle": {spec.key: ("DRAINING" if spec.key in runtime.DRAINING_ACCOUNT_KEYS else "ACTIVE")
                              for spec in runtime.REVIEWED_23_ACCOUNTS}}}
    for spec in runtime.REVIEWED_23_ACCOUNTS:
        current = record
        for key in spec.record_keys:
            current = current.setdefault(key, {})
    record_file = tmp_path / "run_record.json"
    record_file.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(controller, "RUN_RECORD", record_file)
    monkeypatch.setattr(controller, "_verify_running_account", lambda overlay: {})
    monkeypatch.setattr(controller, "load_luna_context", lambda now: SimpleNamespace(status="observed", sources=(SOURCE,), observed_at=NOW))
    monkeypatch.setattr(controller, "load_aggressive_control", lambda *args: control)
    monkeypatch.setattr(controller, "_seen_decision", lambda *args: False)
    monkeypatch.setattr(controller, "_record", lambda row, account: rows.append(deepcopy(row)))
    monkeypatch.setattr(controller, "_api", lambda overlay, route, payload=None:
        routes.append((route, payload)) or ([] if route == "status" else {"total_bot": 10000.} if route == "balance" else {"order_id": "paper-only"}))
    monkeypatch.setattr(paper_news_manual, "decision_rows", lambda path: rows)
    monkeypatch.setattr(context, "load_luna_context", lambda now: SimpleNamespace(status="observed", sources=(SOURCE,), observed_at=NOW))

    assert controller.main(["enter", "--account", "aggressive_vacuum", "--decision-id", approval["decision_id"],
        "--reason", approval["reason"], "--source", SOURCE, "--pair", PAIRS[0], "--side", "short",
        "--stake-pct", ".02", "--leverage", "3", "--reference-rate", "100", "--stop-price", "102",
        "--take-profit-price", "95"]) == 0
    force = [item for item in rows if item.get("action") == "enter"][-1]
    assert force["status"] == "submitted"
    assert routes[-1][0] == "forceenter"
    assert routes[-1][1] == {"pair": PAIRS[0], "side": "short", "stakeamount": 200.,
                             "leverage": 3., "entry_tag": "aggressive_manual:force-no-signal"}

    strategy = PaperAggressiveVacuum(config())
    strategy._control = lambda _now: control
    strategy._current_row = lambda pair, now, **kwargs: pd.Series({
        "date": NOW - timedelta(minutes=5), "ag_signal": False, "ag_score": np.nan,
        "ag_mode": None, "ag_route": "best_guess", "ag_reason": "no automatic setup",
        "ag_atr5": .5, "ag_atr_15m": 1., "ag_pressure_15m": 0.,
        "ag_pressure_change_15m": 0., "ag_relative_volume_15m": 1.,
        "ag_range_expansion_15m": .7, "close_15m": 100., "low_15m": 99.,
        "high_15m": 101., "ag_vah_1h": 101., "ag_val_1h": 99.,
        "ag_thin_above_1h": .3, "ag_thin_below_1h": .3,
        "ag_prior_high_15m": 105., "ag_prior_low_15m": 95.,
        "ag_poc_4h": 100., "ag_vah_4h": 105., "ag_val_4h": 95.,
        "ag_poc_migration_4h": 0., "ag_range_high_1h": 106., "ag_range_low_1h": 94.,
        "ag_range_high_4h": 108., "ag_range_low_4h": 92.,
        "ag_day_range_high_1d": 110., "ag_day_range_low_1d": 90.,
        "ag_cohort_return": 0., "ag_cohort_excess": 0., "ag_cohort_rank": .5,
        "ag_cohort_dispersion": .002, "ag_btc_return45": 0., "ag_eth_return45": 0.,
        "ag_daily_position": .5})
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10000.)
    monkeypatch.setattr("user_data.strategies.paper_aggressive_alt.Trade.get_open_trades", lambda: [])
    entry_tag = routes[-1][1]["entry_tag"]
    leverage = strategy.leverage(PAIRS[0], NOW, 100., 7., 10., entry_tag, "short")
    stake = strategy.custom_stake_amount(PAIRS[0], NOW, 100., 1500., 1., 1500., leverage,
                                          entry_tag, "short")
    assert leverage == 3. and stake == pytest.approx(200.)
    assert strategy.confirm_trade_entry(PAIRS[0], "market", stake * leverage / 100., 100., "gtc",
                                         NOW, entry_tag, "short") is True


def test_fee_meaningful_manual_target_survives_controller_trade_and_order_filled_callback(tmp_path, monkeypatch):
    from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
    from user_data.Custom_Launcher.research import paper_trial_control as controller
    from user_data.Custom_Launcher.research import paper_trial_runtime as runtime
    from user_data.strategies import paper_news_manual

    control = context.AggressiveControl("observed", "manual-target-control", 0, "both", 8., 8., "normal")
    rows, routes, stored = [], [], {}
    record = {"process_recovery": {
        "allowed_accounts": sorted(runtime.REVIEWED_23_ACCOUNT_KEYS),
        "account_lifecycle": {spec.key: ("DRAINING" if spec.key in runtime.DRAINING_ACCOUNT_KEYS else "ACTIVE")
                              for spec in runtime.REVIEWED_23_ACCOUNTS}}}
    for spec in runtime.REVIEWED_23_ACCOUNTS:
        current = record
        for key in spec.record_keys:
            current = current.setdefault(key, {})
    record_file = tmp_path / "run_record.json"
    record_file.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(controller, "RUN_RECORD", record_file)
    monkeypatch.setattr(controller, "_assert_manual_lifecycle_allows", lambda *args: None)
    monkeypatch.setattr(controller, "_verify_running_account", lambda overlay: {})
    monkeypatch.setattr(controller, "load_luna_context",
                        lambda now: SimpleNamespace(status="observed", sources=(SOURCE,), observed_at=NOW))
    monkeypatch.setattr(controller, "load_aggressive_control", lambda *args: control)
    monkeypatch.setattr(controller, "_seen_decision", lambda *args: False)
    monkeypatch.setattr(controller, "_record", lambda row, account: rows.append(deepcopy(row)))
    monkeypatch.setattr(controller, "_api", lambda overlay, route, payload=None:
        routes.append((route, payload)) or ([] if route == "status" else
        {"total_bot": 10000.} if route == "balance" else {"order_id": "paper-only"}))
    monkeypatch.setattr(paper_news_manual, "decision_rows", lambda path: rows)
    monkeypatch.setattr(context, "load_luna_context",
                        lambda now: SimpleNamespace(status="observed", sources=(SOURCE,), observed_at=NOW))
    monkeypatch.setattr(Trade, "get_open_trades", lambda: [])
    monkeypatch.setattr(Trade, "get_custom_data",
                        lambda trade, key, default=None: deepcopy(stored.get((trade.id, key), default)))
    monkeypatch.setattr(Trade, "set_custom_data",
                        lambda trade, key, value: stored.__setitem__((trade.id, key), deepcopy(value)))

    assert controller.main(["enter", "--account", "aggressive_vacuum", "--decision-id", "fee-target-101",
        "--reason", "reviewed 1 percent target with 2 percent stop", "--source", SOURCE,
        "--pair", PAIRS[0], "--side", "long", "--stake-pct", ".015", "--leverage", "3",
        "--reference-rate", "100", "--stop-price", "98", "--take-profit-price", "101"]) == 0
    submitted = [item for item in rows if item.get("action") == "enter"][-1]
    assert submitted["status"] == "submitted"
    assert routes[-1] == ("forceenter", {"pair": PAIRS[0], "side": "long", "stakeamount": 150.,
        "leverage": 3., "entry_tag": "aggressive_manual:fee-target-101"})

    strategy = PaperAggressiveVacuum(config())
    strategy._control = lambda _now: control
    strategy._current_row = lambda pair, now, **kwargs: pd.Series({
        "date": NOW - timedelta(minutes=5), "ag_atr5": 5., "ag_atr_15m": 1.,
        "ag_route": "best_guess", "ag_reason": "no automatic setup"})
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10000.)
    entry_tag = routes[-1][1]["entry_tag"]
    leverage = strategy.leverage(PAIRS[0], NOW, 100., 7., 10., entry_tag, "long")
    stake = strategy.custom_stake_amount(PAIRS[0], NOW, 100., 1000., 1., 1000., leverage,
                                         entry_tag, "long")
    assert leverage == 3. and stake == pytest.approx(150.)
    quantity = stake * leverage / 100.
    assert not strategy.confirm_trade_entry(PAIRS[0], "market", stake * leverage / 100.7,
                                             100.7, "gtc", NOW, entry_tag, "long")
    assert strategy.confirm_trade_entry(PAIRS[0], "market", quantity, 100., "gtc",
                                         NOW, entry_tag, "long")

    trade, order = _freqtrade_trade_order(trade_id=61, pair=PAIRS[0], entry_tag=entry_tag,
        side="long", rate=100., amount=quantity, stake=stake, leverage=leverage)
    strategy_safe_wrapper(strategy.order_filled, supress_error=True)(
        pair=PAIRS[0], trade=trade, order=order, current_time=NOW)
    filled = stored[(trade.id, context.PLAN_KEY)]
    assert filled["target_price"] == 101. and filled["target_kind"] == "main_agent_approved"
    assert filled["stop_price"] == filled["approved_stop_price"] == 98.
    assert filled["fill_stop_adjustment"] is None
    monkeypatch.setattr("user_data.strategies.paper_aggressive_alt.apply_aggressive_protection_update",
                        lambda plan, *args, **kwargs: plan)
    assert strategy.custom_stoploss(PAIRS[0], trade, NOW, 100., 0., False) is not None
    assert stored[(trade.id, context.PLAN_KEY)]["stop_price"] == 98.
    assert strategy.custom_exit(PAIRS[0], trade, NOW, 100., 0.) is None
    assert strategy.custom_exit(PAIRS[0], trade, NOW, 101., .01) == "aggressive_target"


def test_irrecoverable_fill_fault_is_logged_and_fails_closed_under_freqtrade_callback_wrapper(monkeypatch, caplog):
    from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper

    strategy = PaperAggressiveVacuum(config())
    control = context.AggressiveControl("observed", "fault-control", 0, "both", 7., 7., "normal")
    strategy._control = lambda _now: control
    pending = _pending_fill("long")
    pending.update(at=NOW, confirmed=False, control_decision_id=control.decision_id,
                   entry_atr=2., technical_only=False)
    entry_tag = pending["entry_tag"]
    strategy._pending[(PAIRS[0], "long", entry_tag)] = pending
    stored = {}
    monkeypatch.setattr(Trade, "get_custom_data",
                        lambda trade, key, default=None: deepcopy(stored.get((trade.id, key), default)))
    monkeypatch.setattr(Trade, "set_custom_data",
                        lambda trade, key, value: stored.__setitem__((trade.id, key), deepcopy(value)))
    assert strategy.confirm_trade_entry(PAIRS[0], "market", 50., 100., "gtc", NOW,
                                         entry_tag, "long")
    # The accepted quote was 100, but an irrecoverable gap fills exactly at the approved target.
    trade, order = _freqtrade_trade_order(trade_id=62, pair=PAIRS[0], entry_tag=entry_tag,
        side="long", rate=110., amount=50., stake=1100., leverage=5.)
    strategy_safe_wrapper(strategy.order_filled, supress_error=True)(
        pair=PAIRS[0], trade=trade, order=order, current_time=NOW)
    assert (trade.id, context.PLAN_KEY) not in stored
    assert "aggressive_entry_fill_protection_fault" in caplog.text
    assert strategy.custom_stoploss(PAIRS[0], trade, NOW, 110., 0., False) is not None
    assert strategy.custom_exit(PAIRS[0], trade, NOW, 110., 0.) == "aggressive_missing_or_invalid_protection_emergency_exit"


def test_actual_automatic_entry_callbacks_reconcile_and_persist_the_exchange_fill(monkeypatch):
    strategy = PaperAggressiveVacuum(config())
    strategy._control = lambda _now: context.AggressiveControl(
        "observed", "auto-control", 0, "both", 7., 7., "normal")
    strategy._top_rank_allows = lambda pair, now, **kwargs: True
    analyzed = strategy.populate_entry_trend(signal_frame("vacuum", side="long", primary=True),
                                              {"pair": PAIRS[0]})
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (analyzed, None))
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10000.)
    monkeypatch.setattr("user_data.strategies.paper_aggressive_alt.Trade.get_open_trades", lambda: [])
    row = analyzed.iloc[-1]
    assert row["ag_signal"] and row["ag_mode"] == "primary" and row["ag_decision_side"] == "long"
    tag = row["enter_tag"]
    leverage = strategy.leverage(PAIRS[0], NOW, 103., 7., 10., tag, "long")
    stake = strategy.custom_stake_amount(PAIRS[0], NOW, 103., 1500., 1., 1500., leverage, tag, "long")
    assert 3. <= leverage <= 7. and leverage < 7. and 0 < stake <= 1500.
    fill_rate = 103.05
    quantity = stake * leverage / fill_rate
    assert strategy.confirm_trade_entry(PAIRS[0], "market", quantity, fill_rate, "gtc",
                                         NOW, tag, "long")
    saved = {}
    trade = SimpleNamespace(pair=PAIRS[0], is_short=False, entry_side="buy", enter_tag=tag,
        open_rate=fill_rate, amount=quantity, stake_amount=stake, leverage=leverage,
        contract_size=1., id=53,
        get_custom_data=lambda key: saved.get(key),
        set_custom_data=lambda key, value: saved.__setitem__(key, deepcopy(value)))
    strategy.order_filled(PAIRS[0], trade, SimpleNamespace(ft_order_side="buy"), NOW)
    filled = saved[context.PLAN_KEY]
    context.validate_aggressive_filled_plan(filled, actual_pair=PAIRS[0], actual_side="long",
        actual_open_rate=fill_rate, actual_quantity=quantity, actual_stake=stake, actual_leverage=leverage)
    assert filled["provenance"] == "automatic_signal"
    assert filled["quantity"] == pytest.approx(quantity)
    assert filled["open_rate"] == fill_rate and filled["leverage"] == leverage


def _margin_headroom_strategy(monkeypatch, side, equity):
    strategy = PaperAggressiveReclaim(config("aggressive_reclaim"))
    control = context.AggressiveControl("unavailable")
    row = pd.Series({"ag_decision_side": side, "ag_mode": "exploratory", "ag_score": .55,
        "ag_reason": "best-guess regression", "ag_route": "best_guess",
        "ag_atr5": .003807921787002295, "ag_atr_15m": .007795295295912837,
        "ag_prior_low_15m": 1.1169, "ag_prior_high_15m": 1.1402})
    strategy._control = lambda _now: control
    strategy._current_row = lambda pair, now, **kwargs: row
    strategy._geometry = lambda row, direction, rate: (
        rate * (.98 if direction == "long" else 1.02),
        rate * (1.04 if direction == "long" else .96),
        rate, "test_target")
    strategy._context_size_factor = lambda *args: 1.0
    strategy._features = lambda _row: {}
    strategy._top_rank_allows = lambda *args, **kwargs: True
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: equity)
    monkeypatch.setattr(Trade, "get_open_trades", lambda: [])
    return strategy, control, row


def _reclaim_pending_entry(monkeypatch, side):
    strategy = PaperAggressiveReclaim(config("aggressive_reclaim"))
    boundary = 100.
    row = pd.Series({"ag_decision_side": side, "ag_mode": "exploratory", "ag_score": .55,
        "ag_reason": "boundary callback test", "ag_route": "best_guess", "ag_atr5": .02,
        "ag_atr_15m": .4, "ag_prior_low_15m": 100. if side == "long" else 98.,
        "ag_prior_high_15m": 102. if side == "long" else 100.})
    control = context.AggressiveControl("observed", "reclaim-boundary-control", 0, "both", 7., 7., "normal")
    strategy._control = lambda _now: control
    strategy._current_row = lambda pair, now, **kwargs: row
    strategy._geometry = lambda _row, direction, rate: (
        rate * (.98 if direction == "long" else 1.02),
        rate * (1.04 if direction == "long" else .96), rate, "test_target")
    strategy._context_size_factor = lambda *args: 1.
    strategy._features = lambda _row: {}
    strategy._top_rank_allows = lambda *args, **kwargs: True
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10000.)
    monkeypatch.setattr(Trade, "get_open_trades", lambda: [])
    reference_rate = boundary + .001 if side == "long" else boundary - .001
    entry_tag = f"aggressive:reclaim:exploratory:best_guess:{side}"
    stake = strategy.custom_stake_amount(PAIRS[0], NOW, reference_rate, 1500., 1., 1500., 3., entry_tag, side)
    assert stake > 0
    return strategy, row, control, boundary, reference_rate, entry_tag, stake, stake * 3. / reference_rate


def _route_boundary_pending(family, route, side, *, features, provenance="automatic_signal"):
    strategy = CLASSES[family](config(f"aggressive_{family}"))
    control = context.AggressiveControl("observed", "route-boundary-control", 0,
                                        "both", 7., 7., "normal")
    strategy._control = lambda _now: control
    strategy._top_rank_allows = lambda *args, **kwargs: True
    pending = _pending_fill(side)
    mode = "manual" if provenance == "main_agent_force_entry" else (
        "exploratory" if route == "best_guess" else "primary")
    entry_tag = ("aggressive_manual:route-boundary" if mode == "manual" else
                 f"aggressive:{family}:{mode}:{route}:{side}")
    pending.update(at=NOW, family=family, route=route, mode=mode, pair=PAIRS[0], side=side,
        entry_tag=entry_tag, provenance=provenance, reason="route boundary callback test",
        features=dict(features), confirmed=False, control_decision_id=control.decision_id,
        technical_only=control.technical_only)
    strategy._pending[(PAIRS[0], side, entry_tag)] = pending
    amount = pending["stake_usdt"] * pending["requested_leverage"] / pending["reference_rate"]
    return strategy, pending, amount, entry_tag


def _boundary_test_trade(*, side, entry_tag, rate, amount):
    stored = {}
    trade = SimpleNamespace(pair=PAIRS[0], is_short=side == "short",
        entry_side="sell" if side == "short" else "buy", enter_tag=entry_tag,
        open_rate=rate, amount=amount, stake_amount=amount * rate / 5., leverage=5.,
        contract_size=1., id=790, get_custom_data=lambda key: stored.get(key),
        set_custom_data=lambda key, value: stored.__setitem__(key, deepcopy(value)))
    order = SimpleNamespace(ft_order_side=trade.entry_side)
    return trade, order, stored


@pytest.mark.parametrize("family,route,side,feature", OBSERVED_ROUTE_CASES)
def test_observed_route_confirmation_requires_strict_frozen_boundary_side(
        family, route, side, feature):
    boundary = 100.
    strategy, pending, amount, entry_tag = _route_boundary_pending(
        family, route, side, features={feature: boundary})
    valid = boundary + .001 if side == "long" else boundary - .001
    crossed = boundary - .001 if side == "long" else boundary + .001
    for rate, accepted in ((crossed, False), (boundary, False), (valid, True)):
        pending["confirmed"] = False
        assert strategy.confirm_trade_entry(PAIRS[0], "market", amount, rate, "gtc",
                                             NOW, entry_tag, side) is accepted


@pytest.mark.parametrize("family,route,side,feature", OBSERVED_ROUTE_CASES)
def test_observed_route_crossed_market_fill_uses_existing_emergency_protection(
        family, route, side, feature, caplog):
    boundary = 100.
    strategy, _, amount, entry_tag = _route_boundary_pending(
        family, route, side, features={feature: boundary})
    valid = boundary + .001 if side == "long" else boundary - .001
    crossed = boundary - .001 if side == "long" else boundary + .001
    assert strategy.confirm_trade_entry(PAIRS[0], "market", amount, valid, "gtc",
                                         NOW, entry_tag, side)
    trade, order, stored = _boundary_test_trade(
        side=side, entry_tag=entry_tag, rate=crossed, amount=amount)

    with pytest.raises(ValueError, match="crossed its frozen entry boundary"):
        strategy.order_filled(PAIRS[0], trade, order, NOW)
    assert context.PLAN_KEY not in stored
    assert "aggressive_entry_fill_protection_fault" in caplog.text
    assert "follow_up=emergency_stop_and_exit" in caplog.text
    assert strategy.custom_stoploss(PAIRS[0], trade, NOW, crossed, 0., after_fill=False) is not None
    assert strategy.custom_exit(PAIRS[0], trade, NOW, crossed, 0.) == \
        "aggressive_missing_or_invalid_protection_emergency_exit"


def test_observed_route_missing_frozen_boundary_refuses_and_logs_at_confirmation_and_fill(caplog):
    strategy, pending, amount, entry_tag = _route_boundary_pending(
        "vacuum", "profile_acceptance", "long", features={})
    valid = 100.001
    assert not strategy.confirm_trade_entry(PAIRS[0], "market", amount, valid, "gtc",
                                             NOW, entry_tag, "long")
    assert "Missing frozen vah_1h" in caplog.text
    pending["confirmed"] = True
    trade, order, stored = _boundary_test_trade(
        side="long", entry_tag=entry_tag, rate=valid, amount=amount)
    with pytest.raises(ValueError, match="Missing frozen vah_1h"):
        strategy.order_filled(PAIRS[0], trade, order, NOW)
    assert context.PLAN_KEY not in stored
    assert "aggressive_entry_fill_protection_fault" in caplog.text


@pytest.mark.parametrize("family,side,feature", [
    ("vacuum", "long", "vah_1h"), ("vacuum", "short", "val_1h"),
    ("auction", "long", "vah_4h"), ("auction", "short", "val_4h"),
])
def test_automatic_best_guess_stays_allowed_and_is_not_relabelled(family, side, feature):
    boundary = 100.
    strategy, _, amount, entry_tag = _route_boundary_pending(
        family, "best_guess", side, features={feature: boundary})
    crossed = boundary - .001 if side == "long" else boundary + .001
    assert strategy.confirm_trade_entry(PAIRS[0], "market", amount, crossed, "gtc",
                                         NOW, entry_tag, side)
    trade, order, stored = _boundary_test_trade(
        side=side, entry_tag=entry_tag, rate=crossed, amount=amount)
    strategy.order_filled(PAIRS[0], trade, order, NOW)
    assert stored[context.PLAN_KEY]["route"] == "best_guess"


@pytest.mark.parametrize("family,route,side,feature", [
    OBSERVED_ROUTE_CASES[0], OBSERVED_ROUTE_CASES[3],
])
def test_manual_observed_route_force_entry_is_exempt_at_confirmation_and_fill(
        family, route, side, feature):
    boundary = 100.
    strategy, _, amount, entry_tag = _route_boundary_pending(
        family, route, side, features={feature: boundary}, provenance="main_agent_force_entry")
    crossed = boundary - .001 if side == "long" else boundary + .001
    assert strategy.confirm_trade_entry(PAIRS[0], "market", amount, crossed, "gtc",
                                         NOW, entry_tag, side)
    trade, order, stored = _boundary_test_trade(
        side=side, entry_tag=entry_tag, rate=crossed, amount=amount)
    strategy.order_filled(PAIRS[0], trade, order, NOW)
    assert stored[context.PLAN_KEY]["route"] == route
    assert stored[context.PLAN_KEY]["provenance"] == "main_agent_force_entry"


@pytest.mark.parametrize("side", ["long", "short"])
def test_reclaim_confirmation_rejects_crossed_or_touched_frozen_boundary(monkeypatch, side):
    strategy, _, _, boundary, _, entry_tag, _, amount = _reclaim_pending_entry(monkeypatch, side)
    crossed = boundary - .001 if side == "long" else boundary + .001
    valid = boundary + .001 if side == "long" else boundary - .001
    for rate in (crossed, boundary):
        assert not strategy.confirm_trade_entry(PAIRS[0], "market", amount, rate, "gtc", NOW,
                                                 entry_tag, side)
    assert strategy.confirm_trade_entry(PAIRS[0], "market", amount, valid, "gtc", NOW,
                                         entry_tag, side)


@pytest.mark.parametrize("side", ["long", "short"])
def test_reclaim_crossed_market_fill_persists_emergency_stop_and_exit(monkeypatch, tmp_path, side):
    from freqtrade.persistence.models import (
        PairLock, WalletHistory, _CustomData, _KeyValueStoreModel, init_db,
    )
    strategy, _, _, boundary, _, entry_tag, stake, amount = _reclaim_pending_entry(monkeypatch, side)
    valid_confirmation = boundary + .001 if side == "long" else boundary - .001
    crossed_fill = boundary - .001 if side == "long" else boundary + .001
    assert strategy.confirm_trade_entry(PAIRS[0], "market", amount, valid_confirmation, "gtc", NOW,
                                         entry_tag, side)
    classes = (Trade, Order, PairLock, WalletHistory, _CustomData, _KeyValueStoreModel)
    previous = {item: (hasattr(item, "session"), getattr(item, "session", None)) for item in classes}
    init_db(f"sqlite:///{(tmp_path/'reclaim-emergency.sqlite').as_posix()}")
    engine = Trade.session().get_bind()
    try:
        trade, order = _freqtrade_trade_order(trade_id=991 if side == "long" else 992,
            pair=PAIRS[0], entry_tag=entry_tag, side=side, rate=crossed_fill,
            amount=amount, stake=stake, leverage=3.)
        trade.price_precision = 3
        trade.precision_mode_price = 2
        stored = {}
        monkeypatch.setattr(Trade, "get_custom_data",
                            lambda trade, key, default=None: deepcopy(stored.get((trade.id, key), default)))
        monkeypatch.setattr(Trade, "set_custom_data",
                            lambda trade, key, value: stored.__setitem__((trade.id, key), deepcopy(value)))
        Trade.session.add(trade)
        Trade.commit()
        with pytest.raises(ValueError, match="crossed its frozen entry boundary"):
            strategy.order_filled(PAIRS[0], trade, order, NOW)
        assert (trade.id, context.PLAN_KEY) not in stored

        # This is the same regular strategy stoploss path Freqtrade runs after a
        # fill. The invalid plan selects the existing emergency margin stop.
        strategy.ft_stoploss_adjust(crossed_fill, trade, NOW, 0., 0., after_fill=False)
        Trade.commit()
        Trade.session.expire_all()
        persisted = Trade.session.query(Trade).filter_by(id=trade.id).one()
        assert persisted.stop_loss > 0
        emergency_distance = abs(persisted.stop_loss / persisted.open_rate - 1.) * persisted.leverage
        assert emergency_distance <= context.AGGRESSIVE_LIMITS["emergency_margin_loss_pct"] + 1e-6
        assert strategy.custom_exit(PAIRS[0], persisted, NOW, crossed_fill, 0.) == \
            "aggressive_missing_or_invalid_protection_emergency_exit"
    finally:
        Trade.session.remove()
        engine.dispose()
        for item, (had_session, session) in previous.items():
            if had_session:
                item.session = session
            elif hasattr(item, "session"):
                delattr(item, "session")


@pytest.mark.parametrize("side", ["long", "short"])
def test_automatic_market_fill_headroom_reproduces_sui_margin_breach_and_keeps_caps(monkeypatch, side):
    rate, fill_rate = 1.1314, 1.1319
    original_stake = 1468.9101663195
    equity = original_stake / context.AGGRESSIVE_LIMITS["max_margin_pct"]
    margin_cap = equity * context.AGGRESSIVE_LIMITS["max_margin_pct"]
    strategy, control, row = _margin_headroom_strategy(monkeypatch, side, equity)
    entry_tag = f"aggressive:reclaim:exploratory:best_guess:{side}"

    stake = strategy.custom_stake_amount(
        PAIRS[0], NOW, rate, 1500., 1., 1500., 3., entry_tag, side)
    pending = strategy._pending[(PAIRS[0], side, entry_tag)]
    amount = stake * 3. / rate
    upper_fill_rate = rate + .25 * row["ag_atr5"]
    assert stake < original_stake
    assert amount * upper_fill_rate / 3. <= margin_cap + 1e-5

    # Before the headroom reserve, the precise SUI amount/price pair crossed the
    # staged 15% margin cap even though the planned stake itself was under it.
    old_amount = original_stake * 3. / rate
    assert not strategy.confirm_trade_entry(PAIRS[0], "market", old_amount, rate, "gtc",
                                             NOW, entry_tag, side)
    assert strategy.confirm_trade_entry(PAIRS[0], "market", amount, rate, "gtc",
                                         NOW, entry_tag, side)

    # Freqtrade's amount is base quantity; its amount precision rounds this
    # SUI quantity down before the dry-run market fill.
    quantity = math.floor(amount * 10.) / 10.
    old_quantity = math.floor(old_amount * 10.) / 10.
    assert old_quantity * fill_rate / 3. > margin_cap
    filled_stake = quantity * fill_rate / 3.
    assert filled_stake <= margin_cap
    stored = {}
    trade = SimpleNamespace(pair=PAIRS[0], is_short=side == "short",
        entry_side="sell" if side == "short" else "buy", enter_tag=entry_tag,
        open_rate=fill_rate, amount=quantity, stake_amount=filled_stake, leverage=3.,
        contract_size=1., id=71,
        get_custom_data=lambda key, default=None: stored.get(key, default),
        set_custom_data=lambda key, value: stored.__setitem__(key, deepcopy(value)))
    strategy.order_filled(PAIRS[0], trade, SimpleNamespace(ft_order_side=trade.entry_side), NOW)
    plan = stored[context.PLAN_KEY]
    context.validate_aggressive_filled_plan(plan, actual_pair=PAIRS[0],
        actual_side=side, actual_open_rate=fill_rate, actual_quantity=quantity,
        actual_stake=filled_stake, actual_leverage=3.)

    bounded = strategy._automatic_stake_values(
        row=row, side=side, rate=rate, control=control, equity=equity, committed=0.,
        leverage=3., min_stake=1., max_stake=500.)
    assert bounded is not None and bounded["stake"] == pytest.approx(500.)
    assert bounded["stake"] * upper_fill_rate / rate <= margin_cap + 1e-5
    assert strategy._automatic_stake_values(
        row=row, side=side, rate=rate, control=control, equity=equity, committed=0.,
        leverage=3., min_stake=501., max_stake=500.) is None


@pytest.mark.parametrize("side", ["long", "short"])
def test_manual_entry_uses_same_price_headroom_without_changing_approval_interface(monkeypatch, side):
    rate, atr = 1.1314, .003807921787002295
    equity = 10000.
    strategy, control, row = _margin_headroom_strategy(monkeypatch, side, equity)
    control = context.AggressiveControl("observed", "manual-margin-control", 0, "both", 7., 7., "normal")
    strategy._control = lambda _now: control
    approval = {"decision_id": "manual-margin", "account": "paper_aggressive_reclaim",
        "pair": PAIRS[0], "side": side, "reference_rate": rate, "stake_pct": .15,
        "leverage": 3., "stop_price": rate * (.98 if side == "long" else 1.02),
        "take_profit_price": rate * (1.04 if side == "long" else .96),
        "reason": "manual headroom regression", "source": SOURCE,
        "valid_until_utc": (NOW + timedelta(minutes=5)).isoformat(),
        "review_due_at_utc": (NOW + timedelta(hours=4)).isoformat()}
    monkeypatch.setattr("user_data.strategies.paper_aggressive_alt.aggressive_entry_plan",
                        lambda *args, **kwargs: approval)
    entry_tag = "aggressive_manual:manual-margin"
    stake = strategy.custom_stake_amount(PAIRS[0], NOW, rate, 1500., 1., 1500., 3., entry_tag, side)
    pending = strategy._pending[(PAIRS[0], side, entry_tag)]
    amount = stake * 3. / rate
    upper_fill_rate = rate + .15 * atr
    assert stake < equity * context.AGGRESSIVE_LIMITS["max_margin_pct"]
    assert amount * upper_fill_rate / 3. <= equity * context.AGGRESSIVE_LIMITS["max_margin_pct"] + 1e-5
    assert strategy.confirm_trade_entry(PAIRS[0], "market", amount, rate, "gtc", NOW,
                                         entry_tag, side)

    quantity = math.floor(amount * 10.) / 10.
    filled_stake = quantity * upper_fill_rate / 3.
    assert filled_stake <= equity * context.AGGRESSIVE_LIMITS["max_margin_pct"]
    stored = {}
    trade = SimpleNamespace(pair=PAIRS[0], is_short=side == "short",
        entry_side="sell" if side == "short" else "buy", enter_tag=entry_tag,
        open_rate=upper_fill_rate, amount=quantity, stake_amount=filled_stake, leverage=3.,
        contract_size=1., id=72,
        get_custom_data=lambda key, default=None: stored.get(key, default),
        set_custom_data=lambda key, value: stored.__setitem__(key, deepcopy(value)))
    strategy.order_filled(PAIRS[0], trade, SimpleNamespace(ft_order_side=trade.entry_side), NOW)
    context.validate_aggressive_filled_plan(stored[context.PLAN_KEY], actual_pair=PAIRS[0],
        actual_side=side, actual_open_rate=upper_fill_rate, actual_quantity=quantity,
        actual_stake=filled_stake, actual_leverage=3.)
    assert row["ag_atr5"] == pytest.approx(atr)


def test_controller_protection_journal_is_applied_by_actual_strategy_callback(monkeypatch):
    from user_data.Custom_Launcher.research import paper_trial_control as controller
    from user_data.strategies import paper_news_manual

    original = _filled_plan("long")
    trade_row = {"trade_id": 54, "pair": PAIRS[0], "is_short": False,
        "open_rate": 100., "amount": 50., "stake_amount": 1000., "leverage": 5.}
    rows, calls = [], []
    monkeypatch.setattr(controller, "_assert_manual_lifecycle_allows", lambda *args: None)
    monkeypatch.setattr(controller, "_verify_running_account", lambda overlay: {})
    monkeypatch.setattr(controller, "load_luna_context", lambda now: SimpleNamespace(status="observed", sources=(SOURCE,), observed_at=NOW))
    monkeypatch.setattr(controller, "load_aggressive_trade_plan", lambda account, trade_id: original)
    monkeypatch.setattr(controller, "_seen_decision", lambda *args: False)
    monkeypatch.setattr(controller, "_record", lambda row, account: rows.append(deepcopy(row)))
    monkeypatch.setattr(controller, "_api", lambda overlay, route, payload=None:
        calls.append(route) or ([trade_row] if route == "status" else {"total_bot": 10000.}))

    assert controller.main(["protect", "--account", "aggressive_vacuum", "--decision-id", "protect-54",
        "--reason", "tighten after reviewed evidence", "--source", SOURCE,
        "--trade-id", "54", "--stop-price", "98", "--take-profit-price", "110"]) == 0
    approved = rows[-1]
    assert approved["status"] == "recorded_protection" and approved["plan"]["stop_price"] == 98.
    assert calls == ["status", "balance"]
    monkeypatch.setattr(paper_news_manual, "decision_rows", lambda path: rows)

    strategy = PaperAggressiveVacuum(config())
    saved = {}
    trade = SimpleNamespace(pair=PAIRS[0], is_short=False, open_rate=100., amount=50.,
        stake_amount=1000., leverage=5., contract_size=1., id=54,
        get_custom_data=lambda key: original,
        set_custom_data=lambda key, value: saved.__setitem__(key, deepcopy(value)))
    strategy.custom_stoploss(PAIRS[0], trade, NOW, 100., 0., False)
    assert saved[context.PLAN_KEY]["stop_price"] == 98.


@pytest.mark.parametrize("family,route,row,expected", [
    ("vacuum", "profile_acceptance", {"close_15m": 100., "ag_pressure_15m": -.3,
        "ag_vah_1h": 101.}, "vacuum_profile_acceptance_failed"),
    ("reclaim", "sweep_reclaim", {"close_15m": 94., "ag_pressure_15m": 0.,
        "ag_prior_low_15m": 95.}, "reclaim_entry_boundary_lost"),
    ("auction", "accepted_value_escape", {"close_15m": 100., "ag_pressure_15m": 0.,
        "ag_vah_4h": 101.}, "auction_accepted_escape_failed"),
    ("rotation", "cohort_leader_laggard", {"close_15m": 101., "ag_pressure_15m": 0.,
        "ag_cohort_excess": -.001, "ag_cohort_rank": .3}, "rotation_relative_advantage_lost"),
])
def test_actual_custom_exit_has_immediate_target_and_family_failure_routes(monkeypatch, family, route, row, expected):
    strategy_class = CLASSES[family]
    strategy = strategy_class(config(f"aggressive_{family}"))
    plan = dict(_filled_plan("long"), family=family, route=route)
    if family == "vacuum":
        plan["features"] = {"vah_1h": 101.}
        plan["structural_anchor"] = 101.
    elif family == "auction":
        plan["features"] = {"vah_4h": 101.}
        plan["structural_anchor"] = 101.
    if family == "reclaim":
        plan["entry_reclaim_boundary"] = 95.
    trade = SimpleNamespace(pair=PAIRS[0], is_short=False, open_rate=100., amount=50.,
        stake_amount=1000., leverage=5., contract_size=1., id=51,
        open_date_utc=NOW - timedelta(minutes=20), get_custom_data=lambda key: plan,
        set_custom_data=lambda key, value: None)
    monkeypatch.setattr("user_data.strategies.paper_aggressive_alt.apply_aggressive_protection_update",
        lambda *args, **kwargs: plan)

    assert strategy.custom_exit(PAIRS[0], trade, NOW - timedelta(minutes=18), 111., .11) == "aggressive_target"
    current = {"date": NOW - timedelta(minutes=5), "close_15m": 100., "ag_pressure_15m": 0.,
           "ag_vah_1h": 101., "ag_val_1h": 99., "ag_prior_high_15m": 110.,
           "ag_prior_low_15m": 95., "ag_vah_4h": 105., "ag_val_4h": 95.,
           "ag_cohort_excess": .001, "ag_cohort_rank": .8}
    current.update(row)
    strategy._current_row = lambda *args, **kwargs: pd.Series(current)
    trade.open_date_utc = NOW - timedelta(minutes=20)
    assert strategy.custom_exit(PAIRS[0], trade, NOW, 100., 0.) == expected


def test_reclaim_persists_entry_boundary_separate_from_sweep_stop_anchor(monkeypatch):
    strategy = PaperAggressiveReclaim(config("aggressive_reclaim"))
    analyzed = strategy.populate_entry_trend(signal_frame("reclaim", side="long", primary=True),
                                              {"pair": PAIRS[0]})
    row = analyzed.iloc[-1]
    assert row["ag_route"] == "sweep_reclaim"
    strategy._current_row = lambda pair, now, **kwargs: row
    strategy._control = lambda now: context.AggressiveControl(
        "observed", "reclaim-control", 1, "both", 7., 7., "normal")
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10000.)
    monkeypatch.setattr("user_data.strategies.paper_aggressive_alt.Trade.get_open_trades", lambda: [])
    stake = strategy.custom_stake_amount(PAIRS[0], NOW, 101., 1500., 1., 1500., 3.,
                                         row["enter_tag"], "long")
    pending = strategy._pending[(PAIRS[0], "long", row["enter_tag"])]
    quantity = stake * 3. / 101.
    filled = context.build_aggressive_filled_plan(pending, pair=PAIRS[0], side="long",
        open_rate=101., quantity=quantity, stake=stake, leverage=3., contract_size=1., filled_at=NOW)
    assert filled["entry_reclaim_boundary"] == 100.
    assert filled["structural_anchor"] == 99.5
    assert filled["entry_reclaim_boundary"] != filled["structural_anchor"]


@pytest.mark.parametrize("side,boundary,rolling,close,sweep_anchor", [
    ("long", 99., 97., 98., 98.), ("short", 101., 103., 102., 102.)])
def test_reclaim_failure_uses_frozen_entry_boundary_after_normal_hold_and_exempts_manual(
        monkeypatch, side, boundary, rolling, close, sweep_anchor):
    strategy = PaperAggressiveReclaim(config("aggressive_reclaim"))
    plan = _filled_plan(side)
    plan.update(family="reclaim", route="sweep_reclaim", entry_reclaim_boundary=boundary,
                structural_anchor=sweep_anchor)
    current = {"date": NOW - timedelta(minutes=5), "close_15m": close,
        "ag_pressure_15m": 0., "ag_prior_low_15m": rolling, "ag_prior_high_15m": rolling,
        "ag_vah_1h": 101., "ag_val_1h": 99., "ag_cohort_excess": 0., "ag_cohort_rank": .5}
    strategy._current_row = lambda *args, **kwargs: pd.Series(current)
    monkeypatch.setattr("user_data.strategies.paper_aggressive_alt.apply_aggressive_protection_update",
                        lambda existing, *args, **kwargs: existing)
    trade = SimpleNamespace(pair=PAIRS[0], is_short=side == "short", open_rate=100., amount=50.,
        stake_amount=1000., leverage=5., contract_size=1., id=57,
        open_date_utc=NOW - timedelta(minutes=10),
        get_custom_data=lambda key: plan, set_custom_data=lambda key, value: None)
    assert strategy.custom_exit(PAIRS[0], trade, NOW, close, 0.) is None
    assert strategy.custom_stoploss(PAIRS[0], trade, NOW, close, 0., False) is not None
    trade.open_date_utc = NOW - timedelta(minutes=20)
    assert strategy.custom_exit(PAIRS[0], trade, NOW, close, 0.) == "reclaim_entry_boundary_lost"
    manual_plan = dict(plan, provenance="main_agent_force_entry")
    trade.get_custom_data = lambda key: manual_plan
    assert strategy.custom_exit(PAIRS[0], trade, NOW, close, 0.) is None


def test_actual_stoploss_monotonically_persists_cost_adjusted_profit_protection(monkeypatch):
    strategy = PaperAggressiveVacuum(config())
    plan = _filled_plan("long")
    saved = []
    trade = SimpleNamespace(pair=PAIRS[0], is_short=False, open_rate=100., amount=50.,
        stake_amount=1000., leverage=5., contract_size=1., id=52,
        get_custom_data=lambda key: plan,
        set_custom_data=lambda key, value: saved.append(deepcopy(value)))
    monkeypatch.setattr("user_data.strategies.paper_aggressive_alt.apply_aggressive_protection_update",
        lambda *args, **kwargs: plan)
    result = strategy.custom_stoploss(PAIRS[0], trade, NOW, 104., .04, False)
    assert result is not None and saved
    assert saved[-1]["stop_price"] > 100.
    assert saved[-1]["profit_protection"] == "cost_adjusted_0_25R"


def test_aggressive_manual_reduce_is_rejected_before_any_api_write(monkeypatch):
    from user_data.Custom_Launcher.research import paper_trial_control as controller
    monkeypatch.setattr(controller, "_assert_manual_lifecycle_allows", lambda *args: None)
    monkeypatch.setattr(controller, "_verify_running_account", lambda overlay: {})
    monkeypatch.setattr(controller, "load_luna_context", lambda now: SimpleNamespace(status="observed", sources=(SOURCE,), observed_at=NOW))
    routes = []
    monkeypatch.setattr(controller, "_api", lambda overlay, route, payload=None:
        routes.append((route, payload)) or ([{"trade_id": 51, "pair": PAIRS[0], "amount": 50., "is_short": False}]
        if route == "status" else {"total_bot": 10000.}))
    monkeypatch.setattr(controller, "_seen_decision", lambda *args: False)
    monkeypatch.setattr(controller, "_record", lambda *args: None)
    with pytest.raises(ValueError, match="full exits only"):
        controller.main(["reduce", "--account", "aggressive_vacuum", "--decision-id", "reduce-test",
            "--reason", "test rejection", "--source", SOURCE, "--trade-id", "51", "--fraction", ".5"])
    assert [route for route, _ in routes] == ["status", "balance"]


def test_aggressive_controller_keeps_journalled_full_exit_route(monkeypatch):
    from user_data.Custom_Launcher.research import paper_trial_control as controller
    monkeypatch.setattr(controller, "_assert_manual_lifecycle_allows", lambda *args: None)
    monkeypatch.setattr(controller, "_verify_running_account", lambda overlay: {})
    monkeypatch.setattr(controller, "load_luna_context", lambda now: SimpleNamespace(status="observed", sources=(SOURCE,), observed_at=NOW))
    monkeypatch.setattr(controller, "_seen_decision", lambda *args: False)
    journal, calls = [], []
    monkeypatch.setattr(controller, "_record", lambda row, account: journal.append(deepcopy(row)))
    trades = [{"trade_id": 54, "pair": PAIRS[0], "amount": 50., "is_short": False}]
    def api(overlay, route, payload=None):
        calls.append((route, payload))
        if route == "status":
            return trades
        if route == "balance":
            return {"total_bot": 10000.}
        return {"closed": True}
    monkeypatch.setattr(controller, "_api", api)

    assert controller.main(["exit", "--account", "aggressive_vacuum", "--decision-id", "exit-54",
        "--reason", "reviewed full exit", "--source", SOURCE, "--trade-id", "54"]) == 0
    assert calls[-1] == ("forceexit", {"tradeid": "54"})
    assert journal[-1]["status"] == "submitted" and journal[-1]["action"] == "exit"
