from __future__ import annotations

import os
import time
from typing import Any

from binance.client import Client

from bot.utils.config import ExecutionSettings


class BrokerBinance:
    def __init__(self, cfg: ExecutionSettings) -> None:
        api_key = os.getenv(cfg.api_key_env, "")
        api_secret = os.getenv(cfg.api_secret_env, "")
        self.client = Client(api_key, api_secret, testnet=cfg.use_testnet)
        self.cfg = cfg

    def _retry(self, fn, *args, **kwargs):
        for i in range(3):
            try:
                return fn(*args, **kwargs)
            except Exception:
                if i == 2:
                    raise
                time.sleep(2**i)
        return None

    def validate_permissions(self) -> dict[str, Any]:
        return self._retry(self.client.get_account_api_permissions)

    def get_account_state(self) -> dict[str, Any]:
        isolated = self.cfg.margin_mode == "isolated"
        if isolated:
            return self._retry(self.client.get_isolated_margin_account)
        return self._retry(self.client.get_margin_account)

    def place_margin_order(self, symbol: str, side: str, qty: float, side_effect_type: str = "AUTO_BORROW_REPAY") -> dict[str, Any]:
        return self._retry(
            self.client.create_margin_order,
            symbol=symbol,
            side=side,
            type=self.cfg.order_type,
            quantity=round(qty, 6),
            isIsolated="TRUE" if self.cfg.margin_mode == "isolated" else "FALSE",
            sideEffectType=side_effect_type,
        )

    def borrow(self, asset: str, amount: float, is_isolated: bool = True, symbol: str = "BTCUSDT") -> dict[str, Any]:
        return self._retry(
            self.client.create_margin_loan,
            asset=asset,
            amount=amount,
            isIsolated="TRUE" if is_isolated else "FALSE",
            symbol=symbol,
        )

    def repay(self, asset: str, amount: float, is_isolated: bool = True, symbol: str = "BTCUSDT") -> dict[str, Any]:
        return self._retry(
            self.client.repay_margin_loan,
            asset=asset,
            amount=amount,
            isIsolated="TRUE" if is_isolated else "FALSE",
            symbol=symbol,
        )
