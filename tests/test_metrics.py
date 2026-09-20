import numpy as np
import pandas as pd
import pytest

import kairos_factor as kf


def _dates(n):
    return pd.bdate_range("2021-01-01", periods=n)


def _random_panel(n_dates, n_assets, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n_dates, n_assets)),
                        index=_dates(n_dates),
                        columns=[f"S{i}" for i in range(n_assets)])


# ---------------------------------------------------------------------------
# IC / rank IC
# ---------------------------------------------------------------------------

def test_perfect_predictor_rank_ic_is_one():
    fwd = _random_panel(20, 12, seed=1)
    s = kf.rank_ic(fwd, fwd)
    assert np.allclose(s, 1.0, atol=1e-12)
    assert np.allclose(kf.ic(fwd, fwd), 1.0, atol=1e-12)


def test_inverse_predictor_rank_ic_is_minus_one():
    fwd = _random_panel(20, 12, seed=2)
    s = kf.rank_ic(-fwd, fwd)
    assert np.allclose(s, -1.0, atol=1e-12)
    assert np.allclose(kf.ic(-fwd, fwd), -1.0, atol=1e-12)


def test_rank_ic_always_within_bounds():
    f = _random_panel(50, 30, seed=3)
    r = _random_panel(50, 30, seed=4)
    # 混入常数行与重值行，制造并列/退化截面
    f.iloc[5] = 7.0
    r.iloc[9] = r.iloc[9].round()
    s = kf.rank_ic(f, r)
    valid = s.dropna()
    assert ((valid >= -1.0) & (valid <= 1.0)).all()
    # 因子为常数的截面相关系数无定义 -> NaN
    assert np.isnan(s.iloc[5])


def test_ic_handles_missing_pairwise():
    idx = _dates(2)
    f = pd.DataFrame([[1.0, 2.0, 3.0, np.nan]], index=idx[:1], columns=list("abcd"))
    r = pd.DataFrame([[1.0, np.nan, 3.0, 4.0]], index=idx[:1], columns=list("abcd"))
    # 成对有效样本只有 a、c，且完全同序 -> IC = 1
    assert kf.ic(f, r).iloc[0] == pytest.approx(1.0)


def test_ic_summary_known_values():
    s = pd.Series([0.1, 0.2, 0.3], index=_dates(3))
    d = kf.ic_summary(s)
    assert d["count"] == 3
    assert d["mean"] == pytest.approx(0.2)
    assert d["std"] == pytest.approx(0.1)
    assert d["ir"] == pytest.approx(2.0)
    assert d["t_stat"] == pytest.approx(2.0 * np.sqrt(3))
    assert d["positive_ratio"] == pytest.approx(1.0)


def test_ic_summary_edge_cases():
    empty = kf.ic_summary(pd.Series([], dtype="float64"))
    assert empty["count"] == 0 and empty["ir"] == 0.0
    const = kf.ic_summary(pd.Series([0.05] * 10))
    assert const["std"] == pytest.approx(0.0, abs=1e-15)
    assert const["ir"] == 0.0  # 零波动约定返回 0，避免除零
    assert const["t_stat"] == 0.0


# ---------------------------------------------------------------------------
# 分层收益 / 多空
# ---------------------------------------------------------------------------

def test_quantile_returns_monotonic_for_good_factor():
    f = _random_panel(60, 100, seed=5)
    fwd = f * 0.01 + _random_panel(60, 100, seed=6) * 0.001  # 强正相关因子
    qr = kf.quantile_returns(f, fwd, n_quantiles=5)
    assert list(qr.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    means = qr.mean().to_numpy()
    assert np.all(np.diff(means) > 0)  # 分组平均收益随分位单调递增


def test_quantile_returns_exact_when_returns_equal_factor():
    f = _random_panel(10, 25, seed=7)
    qr = kf.quantile_returns(f, f, n_quantiles=5)
    means = qr.mean()
    assert np.all(np.diff(means.to_numpy()) > 0)
    # Q5 - Q1 应等于多空收益
    ls = kf.long_short_returns(f, f, n_quantiles=5)
    pd.testing.assert_series_equal(ls, (qr["Q5"] - qr["Q1"]).rename("long_short"))


def test_long_short_positive_for_predictive_factor():
    f, fwd = kf.make_factor_and_returns(n_dates=200, n_assets=80, true_ic=0.4, seed=8)
    ls = kf.long_short_returns(f, fwd, n_quantiles=5)
    assert ls.mean() > 0
    assert kf.ic_summary(ls)["positive_ratio"] > 0.9


def test_quantile_groups_labels_and_nan():
    f = pd.DataFrame([[5.0, 1.0, 3.0, np.nan]], index=_dates(1), columns=list("abcd"))
    g = kf.quantile_groups(f, n_quantiles=2)
    # 3 个有效值分 2 组：pct 排名 {1/3, 2/3, 1} * 2 向上取整 -> {1, 2, 2}
    assert g.iloc[0, 1] == 1.0   # 因子最低 -> Q1
    assert g.iloc[0, 2] == 2.0
    assert g.iloc[0, 0] == 2.0   # 因子最高 -> Q2
    assert np.isnan(g.iloc[0, 3])
    with pytest.raises(ValueError):
        kf.quantile_groups(f, n_quantiles=1)


# ---------------------------------------------------------------------------
# 覆盖率 / 自相关 / 换手
# ---------------------------------------------------------------------------

def test_factor_coverage():
    f = _random_panel(3, 20, seed=9)
    f.iloc[0, :5] = np.nan
    cov = kf.factor_coverage(f)
    assert cov.iloc[0] == pytest.approx(0.75)
    assert cov.iloc[1] == pytest.approx(1.0)


def test_factor_autocorrelation_persistent_factor():
    base = _random_panel(1, 30, seed=10).iloc[0]
    f = pd.DataFrame(np.tile(base.to_numpy(), (6, 1)), index=_dates(6),
                     columns=base.index)
    ac = kf.factor_autocorrelation(f, lag=1)
    assert np.isnan(ac.iloc[0])
    assert np.allclose(ac.iloc[1:], 1.0, atol=1e-12)
    ac2 = kf.factor_autocorrelation(f, lag=2)
    assert np.isnan(ac2.iloc[:2]).all()
    assert np.allclose(ac2.iloc[2:], 1.0, atol=1e-12)


def test_factor_autocorrelation_of_iid_factor_is_low():
    f = _random_panel(200, 60, seed=11)
    ac = kf.factor_autocorrelation(f).dropna()
    assert abs(ac.mean()) < 0.05


def test_turnover_weights_full_swap_is_one():
    idx = _dates(3)
    w = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], index=idx, columns=["A", "B"])
    to = kf.turnover(w, kind="weights")
    assert np.isnan(to.iloc[0])
    assert to.iloc[1] == pytest.approx(1.0)  # 全部换血
    assert to.iloc[2] == pytest.approx(0.0)  # 不再调仓


def test_turnover_membership():
    idx = _dates(3)
    m = pd.DataFrame([[1.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0]], index=idx, columns=list("ABCD"))
    to = kf.turnover(m, kind="membership")
    assert to.iloc[1] == pytest.approx(1.0)  # 名单完全更换
    assert to.iloc[2] == pytest.approx(0.5)  # 组从 2 只扩到 4 只，换掉一半权重
    with pytest.raises(ValueError):
        kf.turnover(m, kind="unknown")
