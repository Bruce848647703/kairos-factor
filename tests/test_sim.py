import numpy as np
import pandas as pd
import pytest

import kairos_factor as kf


def test_make_factor_and_returns_deterministic():
    a1, r1 = kf.make_factor_and_returns(n_dates=50, n_assets=20, true_ic=0.1, seed=7)
    a2, r2 = kf.make_factor_and_returns(n_dates=50, n_assets=20, true_ic=0.1, seed=7)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(r1, r2)
    b1, _ = kf.make_factor_and_returns(n_dates=50, n_assets=20, true_ic=0.1, seed=8)
    assert not a1.equals(b1)  # 不同 seed 结果不同


def test_make_factor_and_returns_shape_and_alignment():
    f, r = kf.make_factor_and_returns(n_dates=100, n_assets=30, seed=1)
    assert f.shape == (100, 30) and r.shape == (100, 30)
    assert f.index.equals(r.index)
    assert f.columns.equals(r.columns)
    assert f.notna().all().all() and r.notna().all().all()
    # 因子逐截面标准化：均值 0、方差 1
    assert np.allclose(f.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(f.std(axis=1, ddof=0), 1.0, atol=1e-12)


def test_empirical_ic_close_to_true_ic():
    true_ic = 0.3
    f, r = kf.make_factor_and_returns(n_dates=300, n_assets=80, true_ic=true_ic, seed=17)
    assert kf.ic(f, r).mean() == pytest.approx(true_ic, abs=0.05)
    assert kf.rank_ic(f, r).mean() == pytest.approx(true_ic, abs=0.05)


def test_zero_ic_factor_is_uncorrelated():
    f, r = kf.make_factor_and_returns(n_dates=300, n_assets=80, true_ic=0.0, seed=19)
    assert abs(kf.ic(f, r).mean()) < 0.03
    assert abs(kf.rank_ic(f, r).mean()) < 0.03


def test_factor_has_time_persistence():
    f, _ = kf.make_factor_and_returns(n_dates=300, n_assets=80, seed=23,
                                      persistence=0.9)
    ac = kf.factor_autocorrelation(f).dropna().mean()
    assert ac > 0.7


def test_invalid_arguments():
    with pytest.raises(ValueError):
        kf.make_factor_and_returns(true_ic=1.5)
    with pytest.raises(ValueError):
        kf.make_factor_and_returns(n_assets=1)


def test_make_style_exposures():
    labels, cap = kf.make_style_exposures(n_dates=60, n_assets=25, n_industries=4, seed=5)
    assert labels.shape == (60, 25) and cap.shape == (60, 25)
    assert labels.index.equals(cap.index)
    n_ind = labels.iloc[0].nunique()
    assert 1 <= n_ind <= 4
    # 行业归属不随时间变化
    assert (labels.nunique(axis=0) == 1).all()
    assert np.isfinite(cap.to_numpy()).all()
    assert cap.mean().mean() > 5  # 对数市值量级合理（约 10）
