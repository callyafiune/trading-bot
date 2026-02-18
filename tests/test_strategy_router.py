import pandas as pd

from bot.strategy.breakout_atr import BreakoutATRStrategy
from bot.utils.config import StrategyBreakoutSettings, StrategyRouterSettings


def _df_with_signal(regime: str, side: str) -> tuple[pd.DataFrame, int]:
    n = 120
    close = [100.0] * n
    i = 80
    if side == "LONG":
        close[i] = 120.0
    else:
        close[i] = 80.0

    df = pd.DataFrame(
        {
            "high": close,
            "low": close,
            "close": close,
            "regime": [regime] * n,
            "rel_volume_24": [2.0] * n,
            "ma_200": [100.0] * n,
        }
    )
    return df, i


def test_router_allows_only_expected_side() -> None:
    settings = StrategyBreakoutSettings(mode="breakout", breakout_lookback_N=10, use_ma200_filter=False, use_rel_volume_filter=False)
    strategy = BreakoutATRStrategy(settings, StrategyRouterSettings())

    up_short_df, i = _df_with_signal("BULL_TREND", "SHORT")
    sig, reason = strategy.signal_decision(up_short_df, i)
    assert sig is None
    assert reason == "blocked_trend_up_short_only"

    down_long_df, i = _df_with_signal("BEAR_TREND", "LONG")
    sig, reason = strategy.signal_decision(down_long_df, i)
    assert sig is None
    assert reason == "blocked_trend_down_long_only"


def test_short_only_overrides_regime_router() -> None:
    settings = StrategyBreakoutSettings(
        mode="breakout",
        breakout_lookback_N=10,
        use_ma200_filter=False,
        use_rel_volume_filter=False,
        trade_direction="short",
    )
    strategy = BreakoutATRStrategy(settings, StrategyRouterSettings(enable_range=False, enable_chaos=False))
    df, i = _df_with_signal("TRANSITION_RANGE", "SHORT")
    sig, reason = strategy.signal_decision(df, i)
    assert sig is not None
    assert sig.side == "SHORT"
    assert reason is None
