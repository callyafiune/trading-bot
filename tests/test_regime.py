import pandas as pd

from bot.regime.detector import Regime, RegimeDetector
from bot.utils.config import RegimeSettings


def test_regime_trend_case():
    df = pd.DataFrame(
        {
            "realized_vol_24": [0.01] * 50,
            "range_pct": [0.01] * 50,
            "adx_14": [30] * 50,
            "slope_24": [0.01] * 50,
            "bb_width_20": [0.05] * 50,
        }
    )
    reg = RegimeDetector(RegimeSettings()).apply(df)
    assert reg.iloc[-1] == Regime.TREND.value
