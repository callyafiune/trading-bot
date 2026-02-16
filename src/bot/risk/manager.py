from __future__ import annotations

from dataclasses import dataclass

from bot.utils.config import RiskSettings


@dataclass
class PositionSizeResult:
    qty: float
    notional: float
    valid: bool
    reason: str = ""


class RiskManager:
    def __init__(self, settings: RiskSettings) -> None:
        self.settings = settings

    def hit_kill_switch(self, daily_pnl_pct: float, weekly_pnl_pct: float) -> bool:
        return (
            daily_pnl_pct <= -self.settings.daily_loss_limit_pct
            or weekly_pnl_pct <= -self.settings.weekly_loss_limit_pct
        )

    def _approx_liq_price(self, entry: float, side: str, leverage: float) -> float:
        if side == "LONG":
            return entry * (1 - 1 / leverage)
        return entry * (1 + 1 / leverage)

    def size_position(self, equity: float, entry: float, stop: float, side: str) -> PositionSizeResult:
        risk_amount = equity * self.settings.risk_per_trade
        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return PositionSizeResult(0, 0, False, "invalid_stop")

        qty = risk_amount / stop_dist
        notional = qty * entry
        max_notional = equity * self.settings.max_leverage
        if notional > max_notional:
            qty = max_notional / entry
            notional = qty * entry

        leverage = max(notional / equity, 1e-9)
        liq = self._approx_liq_price(entry, side, leverage)
        if side == "LONG":
            buffer_ok = stop > liq * (1 + self.settings.liquidation_buffer_pct)
        else:
            buffer_ok = stop < liq * (1 - self.settings.liquidation_buffer_pct)

        if not buffer_ok:
            for _ in range(20):
                qty *= 0.9
                notional = qty * entry
                leverage = max(notional / equity, 1e-9)
                liq = self._approx_liq_price(entry, side, leverage)
                if side == "LONG":
                    buffer_ok = stop > liq * (1 + self.settings.liquidation_buffer_pct)
                else:
                    buffer_ok = stop < liq * (1 - self.settings.liquidation_buffer_pct)
                if buffer_ok:
                    break

        if qty <= 0 or not buffer_ok:
            return PositionSizeResult(0, 0, False, "liquidation_buffer_failed")
        return PositionSizeResult(qty=qty, notional=notional, valid=True)
