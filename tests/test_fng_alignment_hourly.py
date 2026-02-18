import pandas as pd

from bot.market_data.loader import process_fng_to_1h


def test_fng_alignment_hourly() -> None:
    daily = pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-03T00:00:00Z"],
                utc=True,
            ),
            "fng_value": [40, 70],
        }
    )

    out = process_fng_to_1h(daily, "2025-01-01", "2025-01-04")
    assert len(out) == 72
    assert set(out.columns) == {"ts", "fng_value"}

    jan2 = out[(out["ts"] >= pd.Timestamp("2025-01-02T00:00:00Z")) & (out["ts"] < pd.Timestamp("2025-01-03T00:00:00Z"))]
    assert (jan2["fng_value"] == 40).all()
    jan3 = out[(out["ts"] >= pd.Timestamp("2025-01-03T00:00:00Z")) & (out["ts"] < pd.Timestamp("2025-01-04T00:00:00Z"))]
    assert (jan3["fng_value"] == 70).all()
