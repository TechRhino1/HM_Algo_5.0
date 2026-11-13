"""Bar timestamps must be true UTC, not the broker's EET/EEST wall clock.

Measured on XMGlobal-MT5 (2026-11-14): the weekly gold open is 01:00 broker time
every week except the one between the EU and US DST changes, where it is 00:00 --
i.e. the broker follows the EU DST calendar (UTC+3 summer, UTC+2 winter). Reading
those labels as UTC put every killzone/session gate 2-3 hours early.
"""
import pandas as pd

from jarvis.data.broker_time import broker_bar_times_to_utc
from jarvis.market.sessions import SessionEngine


def _epoch(wall: str) -> int:
    """MT5 'epoch' = broker wall-clock read as if it were UTC."""
    return int(pd.Timestamp(wall, tz="UTC").timestamp())


def test_summer_bar_is_shifted_three_hours():
    out = broker_bar_times_to_utc(pd.Series([_epoch("2026-07-15 10:00")]), live_offset_sec=7200)
    assert out.iloc[0] == pd.Timestamp("2026-07-15 07:00", tz="UTC")


def test_winter_bar_is_shifted_two_hours():
    out = broker_bar_times_to_utc(pd.Series([_epoch("2026-11-10 10:00")]), live_offset_sec=7200)
    assert out.iloc[0] == pd.Timestamp("2026-11-10 08:00", tz="UTC")


def test_unknown_live_offset_still_uses_the_broker_calendar():
    out = broker_bar_times_to_utc(pd.Series([_epoch("2026-07-15 10:00")]), live_offset_sec=0)
    assert out.iloc[0] == pd.Timestamp("2026-07-15 07:00", tz="UTC")


def test_a_broker_on_a_different_clock_uses_its_measured_offset():
    # e.g. an India server at +5:30 -- the EET calendar does not describe it.
    out = broker_bar_times_to_utc(pd.Series([_epoch("2026-07-15 10:00")]), live_offset_sec=19800)
    assert out.iloc[0] == pd.Timestamp("2026-07-15 04:30", tz="UTC")


def test_datetime_input_labelled_utc_is_treated_as_wall_clock():
    legacy = pd.Series(pd.to_datetime(["2026-07-15 10:00"], utc=True))
    out = broker_bar_times_to_utc(legacy, live_offset_sec=7200)
    assert out.iloc[0] == pd.Timestamp("2026-07-15 07:00", tz="UTC")


def test_dst_transition_hours_do_not_raise():
    # EU spring-forward (03:00->04:00 local) and fall-back (04:00 repeats) Sundays: crypto trades through.
    s = pd.Series([_epoch("2026-03-29 03:30"), _epoch("2026-10-25 03:30")])
    out = broker_bar_times_to_utc(s, live_offset_sec=7200)
    assert out.notna().all()


def test_lake_converts_legacy_files_once_and_never_double_converts(tmp_path):
    import json
    import os

    from jarvis.historical.storage import StorageEngine

    st = StorageEngine(base_data_dir=str(tmp_path))
    df = pd.DataFrame({
        "time": pd.date_range("2026-07-15 10:00", periods=3, freq="h", tz="UTC"),
        "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0,
    })
    m = st.write_dataset_atomic("B", "EURUSD", "H1", df)
    # A freshly written file is already UTC: read back unchanged.
    assert st.read_dataset("B", "EURUSD", "H1")["time"].iloc[0] == pd.Timestamp("2026-07-15 10:00", tz="UTC")

    # Simulate a pre-fix file: same data, manifest without the marker -> broker wall clock.
    mpath = os.path.join(st.manifests_dir, f"{m['dataset_id']}.manifest.json")
    legacy = {k: v for k, v in json.load(open(mpath)).items() if k != "time_basis"}
    json.dump(legacy, open(mpath, "w"))
    assert st.read_dataset("B", "EURUSD", "H1")["time"].iloc[0] == pd.Timestamp("2026-07-15 07:00", tz="UTC")


def test_a_07_30_broker_bar_in_winter_is_still_the_asian_session():
    # 07:30 EET in November is 05:30 UTC: before London opens.
    utc = broker_bar_times_to_utc(pd.Series([_epoch("2026-11-10 07:30")]), live_offset_sec=7200).iloc[0]
    assert not SessionEngine.is_forex_killzone_active(utc.to_pydatetime())
