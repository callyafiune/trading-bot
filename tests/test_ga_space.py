from bot.ga.space import ParamSpec, discover_search_space


def test_param_spec_clamp_constraints() -> None:
    breakout = ParamSpec(key="strategy_breakout.breakout_lookback_N", kind="int", low=10, high=240)
    assert breakout.clamp(2) == 10
    assert breakout.clamp(999) == 240

    atr = ParamSpec(key="strategy_breakout.atr_k", kind="float", low=0.1, high=8.0)
    assert atr.clamp(-1.0) == 0.1
    assert atr.clamp(9.5) == 8.0

    left = ParamSpec(key="market_structure.left_bars", kind="int", low=1, high=12)
    assert left.clamp(0) == 1


def test_declared_space_is_authoritative(tmp_path) -> None:
    base = {
        "regime": {"adx_period": 14, "adx_trend_threshold": 28.0},
        "strategy_breakout": {"breakout_lookback_N": 72},
    }
    space_yaml = tmp_path / "ga_space.yaml"
    space_yaml.write_text(
        "parameters:\n"
        "  strategy_breakout.breakout_lookback_N:\n"
        "    type: int\n"
        "    min: 10\n"
        "    max: 120\n",
        encoding="utf-8",
    )
    space = discover_search_space(base, ga_space_path=space_yaml)
    assert "strategy_breakout.breakout_lookback_N" in space.specs
    assert "regime.adx_period" not in space.specs


def test_normalize_genes_enforces_adx_and_funding_constraints(tmp_path) -> None:
    base = {
        "regime": {
            "adx_exit_threshold": 12.0,
            "adx_enter_threshold": 18.0,
            "adx_trend_threshold": 26.0,
        },
        "funding_filter": {
            "z_threshold": 1.0,
            "block_long_if_z_gt": 1.5,
            "block_short_if_z_lt": -1.5,
        },
    }
    space_yaml = tmp_path / "ga_space.yaml"
    space_yaml.write_text(
        "parameters:\n"
        "  regime.adx_exit_threshold:\n"
        "    type: float\n"
        "    min: 8\n"
        "    max: 22\n"
        "  regime.adx_enter_threshold:\n"
        "    type: float\n"
        "    min: 12\n"
        "    max: 28\n"
        "  regime.adx_trend_threshold:\n"
        "    type: float\n"
        "    min: 18\n"
        "    max: 35\n"
        "  funding_filter.z_threshold:\n"
        "    type: float\n"
        "    min: 0.5\n"
        "    max: 3.0\n"
        "  funding_filter.block_long_if_z_gt:\n"
        "    type: float\n"
        "    min: 1.0\n"
        "    max: 4.0\n"
        "  funding_filter.block_short_if_z_lt:\n"
        "    type: float\n"
        "    min: -4.0\n"
        "    max: -1.0\n",
        encoding="utf-8",
    )
    space = discover_search_space(base, ga_space_path=space_yaml)

    genes = space.normalize_genes(
        {
            "regime.adx_exit_threshold": 22.0,
            "regime.adx_enter_threshold": 12.0,
            "regime.adx_trend_threshold": 18.0,
            "funding_filter.z_threshold": 2.5,
            "funding_filter.block_long_if_z_gt": 1.0,
            "funding_filter.block_short_if_z_lt": -1.0,
        }
    )

    assert genes["regime.adx_exit_threshold"] <= genes["regime.adx_enter_threshold"] <= genes["regime.adx_trend_threshold"]
    assert genes["funding_filter.block_long_if_z_gt"] >= genes["funding_filter.z_threshold"]
    assert abs(genes["funding_filter.block_short_if_z_lt"]) >= genes["funding_filter.z_threshold"]
