import numpy as np
import pandas as pd
import pytest

import kairos_factor as kf
from kairos_factor.preprocess import MAD_SCALE, _ndtri


def _dates(n):
    return pd.bdate_range("2021-01-01", periods=n)


# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------

def test_winsorize_mad_clips_outliers():
    row = [1.0, 2.0, 3.0, 4.0, 100.0]
    f = pd.DataFrame([row, row], index=_dates(2), columns=list("abcde"))
    out = kf.winsorize(f, method="mad", n_mad=3.0)
    med, mad = 3.0, 1.0
    upper = med + 3.0 * MAD_SCALE * mad
    lower = med - 3.0 * MAD_SCALE * mad
    # 极端值被裁剪到阈值，普通值保持不变
    assert out.iloc[0].max() == pytest.approx(upper, rel=1e-12)
    assert out.iloc[0, 0] == pytest.approx(1.0)
    # 任意一行都不应再有超过阈值的值
    assert (out.max(axis=1) <= upper + 1e-12).all()
    assert (out.min(axis=1) >= lower - 1e-12).all()


def test_winsorize_quantile_clips_to_bounds():
    vals = np.arange(1.0, 101.0)
    f = pd.DataFrame([vals], index=_dates(1), columns=[f"a{i}" for i in range(100)])
    out = kf.winsorize(f, method="quantile", quantiles=(0.1, 0.9))
    lo = float(f.quantile(0.1, axis=1).iloc[0])
    hi = float(f.quantile(0.9, axis=1).iloc[0])
    assert float(out.min(axis=1).iloc[0]) == pytest.approx(lo, rel=1e-12)
    assert float(out.max(axis=1).iloc[0]) == pytest.approx(hi, rel=1e-12)
    assert ((out >= lo - 1e-12) & (out <= hi + 1e-12)).all().all()


def test_winsorize_preserves_nan_and_validates_method():
    f = pd.DataFrame([[1.0, np.nan, 3.0]], index=_dates(1), columns=list("abc"))
    out = kf.winsorize(f, method="mad")
    assert np.isnan(out.iloc[0, 1])
    with pytest.raises(ValueError):
        kf.winsorize(f, method="unknown")


def test_winsorize_degenerate_mad_row_untouched():
    # 超过一半样本相同 -> MAD=0，该行不做裁剪
    f = pd.DataFrame([[5.0, 5.0, 5.0, 100.0]], index=_dates(1), columns=list("abcd"))
    out = kf.winsorize(f, method="mad")
    assert out.iloc[0, 3] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# zscore / rank_normalize
# ---------------------------------------------------------------------------

def test_zscore_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.normal(size=(10, 50)), index=_dates(10))
    out = kf.zscore(f)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(out.std(axis=1, ddof=1), 1.0, atol=1e-12)


def test_zscore_constant_row_and_nan():
    f = pd.DataFrame([[5.0, 5.0, 5.0], [1.0, np.nan, 3.0]],
                     index=_dates(2), columns=list("abc"))
    out = kf.zscore(f)
    assert list(out.iloc[0]) == [0.0, 0.0, 0.0]          # 常数截面 -> 0
    assert np.isnan(out.iloc[1, 1])                        # 缺失保持 NaN
    assert out.iloc[1, 0] == pytest.approx(-out.iloc[1, 2])  # 对称


def test_rank_normalize_uniform_in_unit_interval():
    f = pd.DataFrame([[10.0, 30.0, 20.0, np.nan]], index=_dates(1), columns=list("abcd"))
    u = kf.rank_normalize(f)
    # 3 个有效值 -> u = (rank-0.5)/3
    assert u.iloc[0, 0] == pytest.approx(0.5 / 3)
    assert u.iloc[0, 2] == pytest.approx(1.5 / 3)
    assert u.iloc[0, 1] == pytest.approx(2.5 / 3)
    assert np.isnan(u.iloc[0, 3])
    assert ((u.dropna(axis=1) > 0) & (u.dropna(axis=1) < 1)).all().all()


def test_rank_normalize_normal_is_symmetric():
    f = pd.DataFrame([np.arange(5.0)], index=_dates(1), columns=list("abcde"))
    z = kf.rank_normalize(f, output="normal")
    row = z.iloc[0].to_numpy()
    assert row[2] == pytest.approx(0.0, abs=1e-12)     # 中位数映射到 0
    assert np.allclose(row, -row[::-1], atol=1e-12)    # 截面对称
    assert abs(row.mean()) < 1e-12


def test_ndtri_known_values():
    p = np.array([0.5, 0.975, 0.025, 0.99, 0.01, np.nan, 0.0, 1.0])
    q = _ndtri(p)
    expected = np.array([0.0, 1.959963985, -1.959963985,
                         2.326347874, -2.326347874, np.nan, -np.inf, np.inf])
    assert np.allclose(q, expected, rtol=0, atol=1e-9, equal_nan=True)


# ---------------------------------------------------------------------------
# fill_cross_section
# ---------------------------------------------------------------------------

def test_fill_cross_section_methods():
    f = pd.DataFrame([[1.0, 2.0, np.nan, 4.0]], index=_dates(1), columns=list("abcd"))
    assert kf.fill_cross_section(f, "median").iloc[0, 2] == pytest.approx(2.0)
    assert kf.fill_cross_section(f, "mean").iloc[0, 2] == pytest.approx(7.0 / 3.0)
    assert kf.fill_cross_section(f, "zero").iloc[0, 2] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        kf.fill_cross_section(f, "ffill")


def test_fill_cross_section_all_nan_row_stays_nan():
    f = pd.DataFrame([[np.nan, np.nan]], index=_dates(1), columns=list("ab"))
    assert kf.fill_cross_section(f, "median").isna().all().all()
    assert kf.fill_cross_section(f, "zero").iloc[0, 0] == 0.0


# ---------------------------------------------------------------------------
# neutralize
# ---------------------------------------------------------------------------

def _neutralize_case(seed=5, n_dates=6, n_assets=40, n_industries=3):
    rng = np.random.default_rng(seed)
    idx = _dates(n_dates)
    cols = [f"S{i}" for i in range(n_assets)]
    cap = pd.DataFrame(rng.normal(10.0, 1.0, (n_dates, n_assets)), index=idx, columns=cols)
    ind_codes = rng.integers(0, n_industries, size=n_assets)
    labels = pd.DataFrame(
        np.tile([f"IND{j}" for j in ind_codes], (n_dates, 1)), index=idx, columns=cols)
    premia = np.array([1.0, -2.0, 3.5])[:n_industries]
    factor = 2.0 * cap + pd.DataFrame(
        np.tile(premia[ind_codes], (n_dates, 1)), index=idx, columns=cols
    ) + rng.normal(0.0, 0.5, (n_dates, n_assets))
    return factor, labels, cap, ind_codes


def test_neutralize_residuals_orthogonal_to_regressors():
    factor, labels, cap, ind_codes = _neutralize_case()
    resid = kf.neutralize(factor, labels, cap)
    assert resid.shape == factor.shape
    for t in range(len(factor)):
        r = resid.iloc[t].to_numpy()
        c = cap.iloc[t].to_numpy()
        # 与对数市值的截面相关性应为 0（OLS 正交性，精确到浮点误差）
        assert abs(np.corrcoef(r, c)[0, 1]) < 1e-8
        # 各行业残差均值应为 0（与哑变量正交）
        for j in np.unique(ind_codes):
            assert abs(r[ind_codes == j].mean()) < 1e-6
        # 残差总体均值也应为 0（截距项）
        assert abs(r.mean()) < 1e-8


def test_neutralize_removes_size_correlation_of_contaminated_factor():
    factor, labels, cap, _ = _neutralize_case()
    contaminated = factor + 0.8 * kf.zscore(cap)
    # 皮尔逊相关性：OLS 正交性保证中性化后精确为 0
    before = kf.ic(contaminated, cap).dropna().abs().mean()
    resid = kf.neutralize(contaminated, labels, cap)
    after = kf.ic(resid, cap).dropna().abs().mean()
    assert before > 0.3
    assert after < 1e-8
    # 秩相关也应从显著水平回到噪声区间
    assert kf.rank_ic(resid, cap).dropna().abs().mean() < 0.15


def test_neutralize_size_only_and_industry_only():
    factor, labels, cap, _ = _neutralize_case()
    r_size = kf.neutralize(factor, None, cap)
    for t in range(len(factor)):
        assert abs(np.corrcoef(r_size.iloc[t], cap.iloc[t])[0, 1]) < 1e-8
    r_ind = kf.neutralize(factor, labels, None)
    assert r_ind.notna().all().all()
    with pytest.raises(ValueError):
        kf.neutralize(factor, None, None)


def test_neutralize_dict_form_membership_panels():
    factor, labels, cap, ind_codes = _neutralize_case()
    dummies = {}
    for j in np.unique(ind_codes):
        member = (labels == f"IND{j}").astype(float)
        dummies[f"IND{j}"] = member
    resid_dict = kf.neutralize(factor, dummies, cap)
    resid_labels = kf.neutralize(factor, labels, cap)
    pd.testing.assert_frame_equal(resid_dict, resid_labels, atol=1e-8, rtol=1e-8)


def test_neutralize_nan_and_insufficient_samples():
    idx = _dates(3)
    cols = [f"S{i}" for i in range(10)]
    rng = np.random.default_rng(1)
    cap = pd.DataFrame(rng.normal(10, 1, (3, 10)), index=idx, columns=cols)
    factor = pd.DataFrame(rng.normal(size=(3, 10)), index=idx, columns=cols)
    # 第 0 行全 NaN；第 1 行只剩 2 个有效样本（< 截距+市值=2 个参数的可估计下限）
    factor.iloc[0] = np.nan
    factor.iloc[1, 2:] = np.nan
    resid = kf.neutralize(factor, None, cap)
    assert resid.iloc[0].isna().all()
    assert resid.iloc[1, 2:].isna().all()
    assert resid.iloc[2].notna().all()  # 正常截面不受影响
