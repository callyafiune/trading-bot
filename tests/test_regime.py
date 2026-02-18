import pandas as pd

from bot.regime.detector import Regime, RegimeDetector
from bot.utils.config import RegimeSettings


def _regime_df() -> pd.DataFrame:
    n = 260
    close = [100 + i * 0.2 for i in range(n)]
    return pd.DataFrame(
        {
            "close": close,
            "ema200": [100.0] * n,
            "slope_ema200": [0.01] * n,
            "vol_roll": [0.01] * n,
            "realized_vol_24": [0.01] * n,
            "atr_pct": [0.01] * n,
            "adx_14": [35.0] * n,
        }
    )


def test_regime_split_ma200() -> None:
    df = _regime_df()
    detector = RegimeDetector(RegimeSettings(adx_trend_threshold=28, macro_confirm_bars=1, micro_confirm_bars=1, chaos_vol_threshold=1.0))

    df_up = df.copy()
    df_up["close"] = df_up["ema200"] + 1.0
    df_up["slope_ema200"] = 0.01
    out_up = detector.apply(df_up)
    assert out_up.iloc[-1].startswith("BULL")

    df_down = df.copy()
    df_down["close"] = df_down["ema200"] - 1.0
    df_down["slope_ema200"] = -0.01
    out_down = detector.apply(df_down)
    assert out_down.iloc[-1].startswith("BEAR")


def test_no_lookahead() -> None:
    df = _regime_df()
    detector = RegimeDetector(RegimeSettings(adx_trend_threshold=28, chaos_vol_percentile=0.8, macro_confirm_bars=1, micro_confirm_bars=1))

    baseline = detector.apply(df)
    mutated = df.copy()
    mutated.loc[mutated.index[-1], "realized_vol_24"] = 100.0
    mutated.loc[mutated.index[-1], "atr_pct"] = 1.0
    changed = detector.apply(mutated)

    assert baseline.iloc[:-1].equals(changed.iloc[:-1])
