import pandas as pd

from bot.regime.detector import Regime, RegimeDetector
from bot.utils.config import RegimeSettings


def _regime_df() -> pd.DataFrame:
    n = 260
    close = [100 + i * 0.2 for i in range(n)]
    return pd.DataFrame(
        {
            "close": close,
            "ma_200": [100.0] * n,
            "realized_vol_24": [0.01] * n,
            "atr_pct": [0.01] * n,
            "adx_14": [35.0] * n,
        }
    )


def test_regime_split_ma200() -> None:
    df = _regime_df()
    detector = RegimeDetector(RegimeSettings(adx_trend_threshold=28))

    df_up = df.copy()
    df_up["close"] = df_up["ma_200"] + 1.0
    out_up = detector.apply(df_up)
    assert out_up.iloc[-1] == Regime.TREND_UP.value

    df_down = df.copy()
    df_down["close"] = df_down["ma_200"] - 1.0
    out_down = detector.apply(df_down)
    assert out_down.iloc[-1] == Regime.TREND_DOWN.value


def test_no_lookahead() -> None:
    df = _regime_df()
    detector = RegimeDetector(RegimeSettings(adx_trend_threshold=28, chaos_vol_percentile=0.8))

    baseline = detector.apply(df)
    mutated = df.copy()
    mutated.loc[mutated.index[-1], "realized_vol_24"] = 100.0
    mutated.loc[mutated.index[-1], "atr_pct"] = 1.0
    changed = detector.apply(mutated)

    assert baseline.iloc[:-1].equals(changed.iloc[:-1])
