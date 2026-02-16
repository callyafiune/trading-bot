from bot.risk.manager import RiskManager
from bot.utils.config import RiskSettings


def test_sizing_respects_max_leverage():
    rm = RiskManager(RiskSettings(max_leverage=2.0))
    out = rm.size_position(1000, entry=100, stop=99, side="LONG")
    assert out.notional <= 2000 + 1e-9
