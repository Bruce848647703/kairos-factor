import numpy as np
import pandas as pd
import pytest

import kairos_factor as kf
from kairos_factor import FactorReport, decay_analysis, forward_returns


def _dates(n):
    return pd.bdate_range("2021-01-01", periods=n)


# ---------------------------------------------------------------------------
# forward_returns
# ---------------------------------------------------------------------------

def test_forward_returns_single_asset_known_values():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=_dates(3))
    fwd1 = forward_returns(prices, horizon=1, kind="price")
    assert fwd1.iloc[0, 0] == pytest.approx(0.10)
    assert fwd1.iloc[1, 0] == pytest.approx(0.10)
    assert np.isnan(fwd1.iloc[2, 0])  # 末尾无未来数据
    fwd2 = forward_returns(prices, horizon=2, kind="price")
    assert fwd2.iloc[0, 0] == pytest.approx(0.21)
    assert np.isnan(fwd2.iloc[1, 0])


def test_forward_returns_price_and_return_paths_agree():
    rng = np.random.default_rng(0)
    prd = pd.DataFrame(rng.normal(0.0, 0.02, (40, 6)), index=_dates(40),
                       columns=list("ABCDEF"))
    prices = 100.0 * (1.0 + prd).cumprod()
    for h in (1, 3, 5):
        a = forward_returns(prices, h, kind="price").to_numpy()
        b = forward_returns(prd, h, kind="return").to_numpy()
        mask = np.isfinite(a) & np.isfinite(b)
        assert mask.sum() > 0
        assert np.allclose(a[mask], b[mask], rtol=1e-10, atol=1e-12)


def test_forward_returns_auto_detects_price_panel():
    rng = np.random.default_rng(1)
    prd = pd.DataFrame(rng.normal(0.0, 0.02, (30, 4)), index=_dates(30),
                       columns=list("ABCD"))
    prices = 100.0 * (1.0 + prd).cumprod()
    auto = forward_returns(prices, 2, kind="auto")
    explicit = forward_returns(prices, 2, kind="price")
    pd.testing.assert_frame_equal(auto, explicit)
    with pytest.raises(ValueError):
        forward_returns(prices, 0)


def test_forward_returns_window_skips_nan():
    prd = pd.DataFrame({"A": [np.nan, 0.1, 0.2, 0.3]}, index=_dates(4))
    fwd2 = forward_returns(prd, horizon=2, kind="return")
    # t=0 的窗口是行 1..2（不含行 0 的 NaN）-> (1.1*1.2)-1
    assert fwd2.iloc[0, 0] == pytest.approx(1.1 * 1.2 - 1.0)


# ---------------------------------------------------------------------------
# decay_analysis
# ---------------------------------------------------------------------------

def _one_period_predictable_case(n_dates=300, n_assets=80, alpha=0.02, seed=21):
    """构造只预测下一期收益的因子：prd[t+1] 含 alpha*f[t]，其余为纯噪声。"""
    rng = np.random.default_rng(seed)
    idx = _dates(n_dates)
    cols = [f"S{i}" for i in range(n_assets)]
    f = pd.DataFrame(rng.normal(size=(n_dates, n_assets)), index=idx, columns=cols)
    prd = pd.DataFrame(rng.normal(0.0, 0.02, (n_dates, n_assets)), index=idx, columns=cols)
    # 用 .to_numpy() 按位置错位相加：prd 行 t+1 含 alpha * f 行 t（仅预测一期）
    prd.iloc[1:] += alpha * f.iloc[:-1].to_numpy()
    return f, prd


def test_decay_analysis_returns_one_row_per_horizon():
    f, prd = _one_period_predictable_case()
    horizons = (1, 2, 5, 10)
    d = decay_analysis(f, prd, horizons=horizons, kind="return")
    assert len(d) == len(horizons)          # 在 horizons 上返回对应长度
    assert list(d.index) == list(horizons)
    for col in ("ic_mean", "rank_ic_mean", "ic_ir", "rank_ic_ir"):
        assert col in d.columns
        assert np.isfinite(d[col]).all()


def test_decay_analysis_ic_decays_with_horizon():
    f, prd = _one_period_predictable_case()
    d = decay_analysis(f, prd, horizons=(1, 2, 3, 5, 10), kind="return")
    means = d["rank_ic_mean"].to_numpy()
    assert np.all(np.diff(means) < 0)       # 单调衰减
    assert means[0] > 0.4                   # h=1 时信号最强（理论值约 0.7）


# ---------------------------------------------------------------------------
# FactorReport
# ---------------------------------------------------------------------------

def test_factor_report_build_and_exports():
    factor, fwd = kf.make_factor_and_returns(
        n_dates=150, n_assets=60, true_ic=0.3, seed=9)
    rep = FactorReport.build(factor, fwd, n_quantiles=5, name="sim_factor")

    assert len(rep.ic) == 150 and len(rep.rank_ic) == 150
    assert list(rep.quantile_returns.columns) == [f"Q{i}" for i in range(1, 6)]
    assert len(rep.long_short) == 150
    assert rep.decay is None  # 未提供价格/收益面板

    d = rep.to_dict()
    assert d["name"] == "sim_factor"
    assert d["n_periods"] == 150 and d["n_assets"] == 60
    assert d["rank_ic_mean"] == pytest.approx(0.3, abs=0.06)
    assert d["long_short_mean"] > 0

    frame = rep.summary_frame()
    assert list(frame.columns) == ["metric", "value"]
    assert len(frame) == len(d)


def test_factor_report_with_decay():
    # persistence=0：因子逐期独立、只预测下一期收益 -> IC 随期限单调衰减
    factor, fwd = kf.make_factor_and_returns(n_dates=200, n_assets=40, true_ic=0.2,
                                             seed=13, persistence=0.0)
    prd = fwd.shift(1)  # 期间收益 = 远期收益滞后一行
    horizons = (1, 2, 5)
    rep = FactorReport.build(factor, fwd, prices_or_returns=prd,
                             horizons=horizons, kind="return")
    assert rep.decay is not None
    assert list(rep.decay.index) == list(horizons)
    assert rep.decay.loc[1, "rank_ic_mean"] > rep.decay.loc[5, "rank_ic_mean"]
