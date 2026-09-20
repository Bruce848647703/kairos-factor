"""因子预处理模块。

提供逐截面（按日期行）进行的常用因子变换：
- winsorize:         去极值（MAD 法 / 分位数法）
- zscore:            截面标准化（均值 0、标准差 1）
- rank_normalize:    截面排名映射到 (0,1) 均匀分布或正态分位
- fill_cross_section: 截面缺失值填充（中位数 / 均值 / 零）
- neutralize:        行业 + 市值中性化（OLS 回归取残差）

约定：因子面板均为 pandas.DataFrame，index=日期，columns=资产；
所有变换只使用当期截面信息，不引入任何跨期（未来）数据。
纯 numpy/pandas 实现，正态分位映射自带 numpy 回退，不依赖 scipy。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

# 正态假设下 MAD 与标准差的换算系数：1 / Phi^{-1}(0.75) ≈ 1.4826
MAD_SCALE = 1.4826

Panel = pd.DataFrame
LabelsLike = Union[pd.DataFrame, Dict[str, pd.DataFrame], None]


def _as_frame(panel: Panel) -> Panel:
    """统一转成 float64 的 DataFrame，避免就地修改调用方数据。"""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("输入必须是 pandas.DataFrame（index=日期, columns=资产）")
    return panel.astype("float64")


# ---------------------------------------------------------------------------
# 去极值
# ---------------------------------------------------------------------------

def winsorize(factor: Panel, method: str = "mad", n_mad: float = 3.0,
              quantiles: Sequence[float] = (0.01, 0.99)) -> Panel:
    """逐截面去极值（缩尾处理）。

    参数
    ----
    method:    "mad"      —— 以截面中位数为中心，裁剪到中位数 ± n_mad * 1.4826 * MAD；
               "quantile" —— 裁剪到截面分位数 [q_low, q_high]。
    n_mad:     MAD 法的倍数阈值（默认 3 倍，约等价于正态分布 3 sigma）。
    quantiles: 分位数法的 (下分位, 上分位)，默认 (0.01, 0.99)。

    返回裁剪后的新面板，NaN 位置保持不变。
    MAD 为 0 的退化截面（超过一半样本取值相同）不做裁剪，避免整行被压成常数。
    """
    f = _as_frame(factor)
    if method == "mad":
        med = f.median(axis=1)
        mad = f.sub(med, axis=0).abs().median(axis=1)
        scaled = MAD_SCALE * mad * float(n_mad)
        # 退化截面（MAD=0）阈值置为无穷，即不裁剪
        upper = (med + scaled).where(scaled > 0, np.inf)
        lower = (med - scaled).where(scaled > 0, -np.inf)
        return f.clip(lower=lower, upper=upper, axis=0)
    if method == "quantile":
        q_low, q_high = sorted(float(q) for q in quantiles)
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError("quantiles 必须满足 0 <= q_low < q_high <= 1")
        lower = f.quantile(q_low, axis=1)
        upper = f.quantile(q_high, axis=1)
        return f.clip(lower=lower, upper=upper, axis=0)
    raise ValueError(f"未知去极值方法: {method!r}，仅支持 'mad' / 'quantile'")


# ---------------------------------------------------------------------------
# 标准化 / 排名
# ---------------------------------------------------------------------------

def zscore(factor: Panel, ddof: int = 1) -> Panel:
    """逐截面标准化：(x - 均值) / 标准差。

    标准差为 0 的常数截面输出 0（而非 NaN）；因子缺失位置保持 NaN。
    """
    f = _as_frame(factor)
    mu = f.mean(axis=1)
    sd = f.std(axis=1, ddof=ddof)
    sd_safe = sd.where(sd > 1e-14)
    out = f.sub(mu, axis=0).div(sd_safe, axis=0)
    # 有效值但因零标准差变成 NaN 的位置，回退为 0
    degenerate = f.notna() & out.isna()
    return out.mask(degenerate, 0.0)


def rank_normalize(factor: Panel, output: str = "uniform",
                   method: str = "average") -> Panel:
    """逐截面排名归一化。

    先把每行有效值映射为 u = (rank - 0.5) / n（n 为该行有效个数），保证 u 严格落在 (0,1)
    且关于截面对称：
    - output="uniform": 直接返回 u（均匀分布映射）；
    - output="normal":  返回 Phi^{-1}(u)（正态分位映射，截面均值为 0、近似单位方差）。

    正态分位用内置 _ndtri（有理逼近 + 牛顿精化）计算，无需 scipy。
    NaN 位置保持不变；method 为 pandas rank 的并列处理方式。
    """
    f = _as_frame(factor)
    if output not in ("uniform", "normal"):
        raise ValueError(f"未知输出类型: {output!r}，仅支持 'uniform' / 'normal'")
    n = f.notna().sum(axis=1).replace(0, np.nan)
    r = f.rank(axis=1, method=method)
    u = r.sub(0.5).div(n, axis=0)
    if output == "uniform":
        return u
    values = _ndtri(u.to_numpy())
    return pd.DataFrame(values, index=u.index, columns=u.columns).where(u.notna())


def fill_cross_section(factor: Panel, method: str = "median") -> Panel:
    """逐截面填充缺失值。

    method ∈ {"median", "mean", "zero"}：分别用当期截面中位数、均值或 0 填充。
    整行全为 NaN 时，median/mean 无法给出统计量，该行保持 NaN。
    """
    f = _as_frame(factor)
    if method == "zero":
        return f.fillna(0.0)
    if method == "median":
        stat = f.median(axis=1)
    elif method == "mean":
        stat = f.mean(axis=1)
    else:
        raise ValueError(f"未知填充方法: {method!r}，仅支持 'median' / 'mean' / 'zero'")
    # 转置后按列对齐填充，实现「每行用自己的截面统计量」
    return f.T.fillna(stat).T


# ---------------------------------------------------------------------------
# 正态分位函数（numpy 自研实现）
# ---------------------------------------------------------------------------

# 有理逼近系数（Acklam 型），随后用牛顿迭代精化到机器精度，
# 因此即使系数存在尾差也不影响最终结果。
_ACKLAM_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACKLAM_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01)
_ACKLAM_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACKLAM_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00)
_TAIL_P = 0.02425
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
_ERF = np.frompyfunc(math.erf, 1, 1)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """标准正态 CDF，基于 math.erf 的向量化实现。"""
    return 0.5 * (1.0 + _ERF(x / _SQRT2).astype(np.float64))


def _rational_approx(p: np.ndarray) -> np.ndarray:
    """正态分位的有理逼近初值（中部 + 双尾三段）。"""
    x = np.empty_like(p)
    lo = p < _TAIL_P
    hi = p > 1.0 - _TAIL_P
    mid = ~(lo | hi)

    q = p[mid] - 0.5
    r = q * q
    num = (((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r
             + _ACKLAM_A[3]) * r + _ACKLAM_A[4]) * r + _ACKLAM_A[5]) * q
    den = ((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r
            + _ACKLAM_B[3]) * r + _ACKLAM_B[4]) * r + 1.0
    x[mid] = num / den

    for mask, sign in ((lo, 1.0), (hi, -1.0)):
        if not mask.any():
            continue
        tail = p[mask] if sign > 0 else 1.0 - p[mask]
        q = np.sqrt(-2.0 * np.log(tail))
        num = (((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q
                 + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q + _ACKLAM_C[5])
        den = (((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q
               + _ACKLAM_D[3]) * q + 1.0
        x[mask] = sign * num / den
    return x


def _ndtri(p: np.ndarray) -> np.ndarray:
    """标准正态分位函数 Phi^{-1}(p)，纯 numpy 实现。

    先做有理逼近，再用两次牛顿迭代 x <- x + (p - Phi(x)) / phi(x) 精化到机器精度。
    p=0 / p=1 分别返回 -inf / +inf，NaN 与越界值返回 NaN。
    """
    p = np.asarray(p, dtype="float64")
    out = np.full(p.shape, np.nan)
    out[p == 0.0] = -np.inf
    out[p == 1.0] = np.inf
    ok = np.isfinite(p) & (p > 0.0) & (p < 1.0)
    if not ok.any():
        return out
    q = p[ok]
    x = _rational_approx(q)
    for _ in range(2):
        pdf = np.exp(-0.5 * x * x) / _SQRT2PI
        x = x + (q - _norm_cdf(x)) / pdf
    out[ok] = x
    return out


# ---------------------------------------------------------------------------
# 中性化
# ---------------------------------------------------------------------------

def _industry_matrix(industry_dummies: LabelsLike, date, assets: pd.Index
                     ) -> Optional[np.ndarray]:
    """取出某截面、按 assets 顺序排列的行业哑变量矩阵 (n_assets, n_industries)。

    支持两种输入形态：
    - dict：{行业名 -> 成员面板(0/1/bool，index=日期, columns=资产)}；
    - DataFrame：行业标签面板（index=日期, columns=资产，值为行业代码/名称）。
    无法归属行业的资产对应行填 NaN（随后被视作无效样本剔除）。
    """
    if industry_dummies is None:
        return None
    if isinstance(industry_dummies, dict):
        cols = []
        for name in sorted(industry_dummies):
            panel = industry_dummies[name]
            row = panel.reindex(index=[date], columns=assets).iloc[0]
            cols.append(row.to_numpy(dtype="float64", copy=True))
        if not cols:
            return None
        mat = np.column_stack(cols)
        # 非 0/1 的成员标记统一成 0/1
        return np.where(np.isnan(mat), np.nan, (mat != 0).astype("float64"))

    if isinstance(industry_dummies, pd.DataFrame):
        labels = industry_dummies.reindex(index=[date], columns=assets).iloc[0]
        known = labels.notna().to_numpy()
        codes, uniques = pd.factorize(labels[known])
        if len(uniques) == 0:
            return None
        mat = np.zeros((len(assets), len(uniques)))
        mat[np.flatnonzero(known), codes] = 1.0
        mat[~known, :] = np.nan  # 行业未知的资产视作无效样本
        return mat

    raise TypeError("industry_dummies 必须是标签 DataFrame 或 {行业名: 成员面板} 字典")


def neutralize(factor: Panel, industry_dummies: LabelsLike = None,
               log_mktcap: Optional[Panel] = None) -> Panel:
    """行业 + 市值中性化：逐截面 OLS 回归取残差。

    对每个日期，把因子值 y 对回归元 X = [截距, 行业哑变量, 对数市值] 做最小二乘，
    用 numpy.linalg.lstsq 求解并返回残差 y - X @ beta。
    截距与全量行业哑变量存在共线性，lstsq 给出最小范数解，但残差（正交投影）唯一。

    参数
    ----
    industry_dummies: 行业标签面板，或 {行业名 -> 成员面板} 字典；None 表示不做行业中性。
    log_mktcap:       对数流通市值面板；None 表示不做市值中性。

    样本有效性：因子、市值、行业归属任一缺失的资产，该截面残差为 NaN。
    若某截面有效样本数 <= 回归元个数，无法稳健估计，该截面整体返回 NaN
    （避免欠定方程组产生虚假的全零残差）。
    """
    if industry_dummies is None and log_mktcap is None:
        raise ValueError("industry_dummies 与 log_mktcap 至少提供一个")
    f = _as_frame(factor)
    cap = _as_frame(log_mktcap).reindex(index=f.index, columns=f.columns) if log_mktcap is not None else None

    out = pd.DataFrame(np.nan, index=f.index, columns=f.columns, dtype="float64")
    assets = f.columns
    for date in f.index:
        y = f.loc[date].to_numpy(dtype="float64")
        blocks = [np.ones(len(assets))]
        ind = _industry_matrix(industry_dummies, date, assets)
        if ind is not None:
            blocks.append(ind)
        cap_row = None
        if cap is not None:
            cap_row = cap.loc[date].to_numpy(dtype="float64")
            blocks.append(cap_row.reshape(-1, 1))
        X = np.column_stack(blocks)

        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
        n_valid = int(valid.sum())
        if n_valid <= X.shape[1]:
            continue  # 样本不足，该截面保持 NaN
        beta, _, _, _ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
        out.loc[date, out.columns[valid]] = y[valid] - X[valid] @ beta
    return out
