"""Read-only briefing guards with mocked market data, never live orders."""
from datetime import datetime, timezone
import io
import json

import pytest

from user_data.Custom_Launcher.research import paper_trial_snapshot as snapshot


def test_higher_snapshot_excludes_open_candles_and_bounds_requests(monkeypatch):
    now=datetime(2026,9,29,12,tzinfo=timezone.utc)
    closed=int(now.timestamp()*1000)-1
    requests=[]
    def fetch(url,timeout):
        requests.append((url,timeout))
        return io.BytesIO(json.dumps([[0,"100","110","90","105","1",closed-1000],
            [1,"105","120","95","110","2",closed],
            [2,"110","999","1","900","3",closed+1000]]).encode())
    monkeypatch.setattr(snapshot,"urlopen",fetch)
    rows=snapshot.higher_snapshot(now)
    assert len(requests)==len(rows)==4
    assert all(timeout==12 and "limit=9" in url for url,timeout in requests)
    assert all(row["close"]==110 and row["recent_high"]==120 and row["recent_low"]==90 for row in rows)
    assert {row["recent_completed_hours"] for row in rows}=={8,48}


def test_higher_snapshot_does_not_invent_missing_history(monkeypatch):
    monkeypatch.setattr(snapshot,"urlopen",lambda *a,**k:io.BytesIO(b"[]"))
    with pytest.raises(ValueError,match="Insufficient completed"):
        snapshot.higher_snapshot(datetime.now(timezone.utc))


def test_no_flags_means_no_network_or_dataset_scan(monkeypatch,capsys):
    monkeypatch.setattr("sys.argv",["paper_trial_snapshot"])
    monkeypatch.setattr(snapshot,"urlopen",lambda *a,**k:pytest.fail("Unexpected online call"))
    monkeypatch.setattr(snapshot,"source_snapshot",lambda *a:pytest.fail("Unexpected archive scan"))
    monkeypatch.setattr(snapshot,"account_snapshot",lambda *a:pytest.fail("Unexpected account access"))
    snapshot.main()
    assert set(json.loads(capsys.readouterr().out))=={"observed_at_utc"}
