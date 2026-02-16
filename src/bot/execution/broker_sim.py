from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fill:
    price: float
    qty: float
    fee: float


class BrokerSim:
    def __init__(self, fee_rate: float, slippage_rate: float) -> None:
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def execute_entry(self, side: str, open_price: float, qty: float) -> Fill:
        price = open_price * (1 + self.slippage_rate) if side == "LONG" else open_price * (1 - self.slippage_rate)
        fee = price * qty * self.fee_rate
        return Fill(price, qty, fee)

    def execute_exit(self, side: str, price: float, qty: float) -> Fill:
        exec_price = price * (1 - self.slippage_rate) if side == "LONG" else price * (1 + self.slippage_rate)
        fee = exec_price * qty * self.fee_rate
        return Fill(exec_price, qty, fee)
