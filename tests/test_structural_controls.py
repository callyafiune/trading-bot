import pandas as pd

from bot.backtest.engine import BacktestEngine
from bot.cli import _attach_mtf_features
from bot.strategy.breakout_atr import BreakoutATRStrategy
from bot.utils.config import MultiTimeframeSettings, StrategyBreakoutSettings, StrategyRouterSettings, load_settings


def _df_for_breakout(close_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=len(close_values), freq="h", tz="UTC"),
            "open": close_values,
            "high": [c + 0.5 for c in close_values],
            "low": [c - 0.5 for c in close_values],
            "close": close_values,
            "volume": [1000.0] * len(close_values),
            "atr_14": [1.0] * len(close_values),
            "regime": ["BULL_TREND"] * len(close_values),
            "rel_volume_24": [2.0] * len(close_values),
        }
    )


def test_mtf_4h_alignment_to_1h_no_lookahead() -> None:
    df = _df_for_breakout([100.0 + i for i in range(12)])
    cfg = load_settings()
    cfg.multi_timeframe.enabled = True
    cfg.multi_timeframe.ma_period = 2

    out = _attach_mtf_features(df, cfg)
    assert pd.isna(out.loc[out.index[3], "close_4h"])
    assert out.loc[out.index[4], "close_4h"] == out.loc[out.index[3], "close"]


def test_mtf_requires_negative_slope_for_short() -> None:
    df = _df_for_breakout([100, 99, 98, 97, 96, 95])
    df["regime"] = "BEAR_TREND"
    df["close_4h"] = 90.0
    df["ema_200_4h"] = 95.0
    df["ema_slope_4h"] = -0.1
    df.loc[df.index[2], ["ema_12", "ema_26"]] = [101.0, 100.0]
    df.loc[df.index[3], ["ema_12", "ema_26"]] = [98.0, 99.0]

    strategy = BreakoutATRStrategy(
        StrategyBreakoutSettings(
            mode="ema",
            use_ma200_filter=False,
            use_rel_volume_filter=False,
        ),
        router_settings=StrategyRouterSettings(),
        mtf_settings=MultiTimeframeSettings(enabled=True, require_trend_alignment=True),
    )
    sig_ok, reason_ok = strategy.signal_decision(df, 3)
    assert reason_ok is None
    assert sig_ok is not None and sig_ok.side == "SHORT"

    df["ema_slope_4h"] = 0.1
    sig_blocked, reason_blocked = strategy.signal_decision(df, 3)
    assert sig_blocked is None
    assert reason_blocked == "mtf"


def test_time_exit_hard_trigger() -> None:
    cfg = load_settings()
    cfg.multi_timeframe.enabled = False
    cfg.strategy_breakout.use_ma200_filter = False
    cfg.strategy_breakout.use_rel_volume_filter = False
    cfg.strategy_breakout.breakout_lookback_N = 2
    cfg.strategy_breakout.trade_direction = "short"
    cfg.strategy_router.overrides.bear_trend.breakout_N = 2
    cfg.adaptive_trailing.enabled = False
    cfg.time_exit.enabled = True
    cfg.time_exit.max_holding_hours = 2
    cfg.time_exit.soft_exit_hours = 50
    cfg.time_exit.min_r_multiple_after_soft = 100.0

    df = _df_for_breakout([120, 119, 118, 110, 109, 108, 107, 106, 105, 104])
    df["regime"] = "BEAR_TREND"
    df["high"] = [c + 0.2 for c in df["close"]]
    df["low"] = [c - 0.2 for c in df["close"]]
    trades, _ = BacktestEngine(cfg).run(df)
    assert not trades.empty
    assert trades.iloc[0]["exit_reason"] == "time_stop"
    assert int(trades.iloc[0]["holding_hours"]) >= 2


def test_adaptive_trailing_short_is_monotonic() -> None:
    cfg = load_settings()
    cfg.multi_timeframe.enabled = False
    cfg.time_exit.enabled = False
    cfg.strategy_breakout.use_ma200_filter = False
    cfg.strategy_breakout.use_rel_volume_filter = False
    cfg.strategy_breakout.breakout_lookback_N = 2
    cfg.strategy_breakout.trade_direction = "short"
    cfg.strategy_router.overrides.bear_trend.breakout_N = 2
    cfg.adaptive_trailing.enabled = True
    cfg.adaptive_trailing.activate_after_R = 0.5
    cfg.adaptive_trailing.trailing_atr_multiplier = 1.5

    close = [100, 99, 98, 90, 88, 86, 84, 85, 86, 87]
    df = _df_for_breakout(close)
    df["regime"] = "BEAR_TREND"
    df["high"] = [c + 0.5 for c in close]
    df["low"] = [c - 0.5 for c in close]
    df.loc[df.index[6], "high"] = 84.5
    df.loc[df.index[7], "high"] = 86.0

    trades, _ = BacktestEngine(cfg).run(df)
    assert not trades.empty
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "trailing_stop"
    assert float(trade["stop_final"]) <= float(trade["stop_init"])
