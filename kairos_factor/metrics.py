"""因子评价指标模块。

提供 IC 序列、IC 汇总统计、分层（分位）收益、多空收益、覆盖率、
因子自相关与换手率等常用因子评价工具。

约定：
- 因子面板与远期收益面板均为 DataFrame（index=日期, columns=资产），
  且同一行代表「当期因子」与「当期之后实现的远期收益」，天然防未来函数；
- 所有相关性均为截面（按行）计算，缺失值按成对剔除处理；
- 返回的序列均以日期为 index，样本不足的截面为 NaN。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

Panel = pd.DataFrame


def _as_frame(panel: Panel) -> Panel:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("输入必须是 pandas.DataFrame（index=日期, columns=资产）")
    return panel.astype("float64")


def _align_pair(factor: Panel, forward_returns: Panel
                ) -> Tuple[Panel, Panel, pd.DataFrame]:
    """把两个面板对齐到共同的日期/资产，并按「成对有效」掩码剔除缺失。"""
    f = _as_frame(factor)
    r = _as_frame(forward_returns)
    idx = f.index.intersection(r.index)
    cols = f.columns.intersection(r.columns)
    f = f.loc[idx, cols]
    r = r.loc[idx, cols]
    mask = f.notna() & r.notna()
    return f.where(mask), r.where(mask), mask


def _row_corr(f: Panel, r: Panel, mask: pd.DataFrame, min_obs: int,
              spearman: bool) -> pd.Series:
    """逐截面相关系数的向量化实现（spearman=True 时先做截面排名）。"""
    if spearman:
        f = f.rank(axis=1)
        r = r.rank(axis=1)
    fm = f.sub(f.mean(axis=1), axis=0)
    rm = r.sub(r.mean(axis=1), axis=0)
    num = (fm * rm).sum(axis=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    corr = num / den.where(den > 1e-14)
    n_obs = mask.sum(axis=1)
    corr = corr.where(n_obs >= min_obs)
    # 数学上相关系数必在 [-1,1]，clip 消除浮点尾差
    return corr.clip(-1.0, 1.0)


def ic(factor: Panel, forward_returns: Panel, min_obs: int = 2) -> pd.Series:
    """逐期截面皮尔逊 IC 序列。min_obs 为参与计算的最少成对样本数。"""
    f, r, mask = _align_pair(factor, forward_returns)
    return _row_corr(f, r, mask, min_obs, spearman=False).rename("ic")


def rank_ic(factor: Panel, forward_returns: Panel, min_obs: int = 2) -> pd.Series:
    """逐期截面斯皮尔曼秩 IC 序列（对因子与收益先做截面排名再求皮尔逊相关）。"""
    f, r, mask = _align_pair(factor, forward_returns)
    return _row_corr(f, r, mask, min_obs, spearman=True).rename("rank_ic")


def ic_summary(ic_series: pd.Series, ddof: int = 1) -> Dict[str, float]:
    """IC 序列的汇总统计。

    返回 dict：count（非 NaN 期数）、mean、std、ir（= mean / std）、
    t_stat（= mean / std * sqrt(count)）、positive_ratio（IC>0 占比）。
    约定：序列为空时 mean/std 为 NaN；std 为 0 时 ir/t_stat 返回 0（避免除零）。
    """
    s = pd.Series(ic_series, dtype="float64").dropna()
    n = int(len(s))
    if n == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"),
                "ir": 0.0, "t_stat": 0.0, "positive_ratio": float("nan")}
    mean = float(s.mean())
    std = float(s.std(ddof=ddof)) if n > 1 else 0.0
    if std > 1e-14:
        ir = mean / std
        t_stat = ir * float(np.sqrt(n))
    else:
        ir, t_stat = 0.0, 0.0
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "ir": float(ir),
        "t_stat": float(t_stat),
        "positive_ratio": float((s > 0).sum() / n),
    }


def quantile_groups(factor: Panel, n_quantiles: int = 5) -> Panel:
    """逐截面按因子值把资产分成 n_quantiles 组。

    返回组号面板：1 = 因子值最低组，n_quantiles = 因子值最高组；
    因子缺失处为 NaN。组内并列值按列顺序稳定切分（rank method="first"）。
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles 必须 >= 2")
    f = _as_frame(factor)
    pct_rank = f.rank(axis=1, method="first", pct=True)
    return np.ceil(pct_rank * n_quantiles).clip(1.0, float(n_quantiles))


def quantile_returns(factor: Panel, forward_returns: Panel,
                     n_quantiles: int = 5) -> Panel:
    """分层（分位）组合的逐期等权平均远期收益。

    每个截面按因子值分成 n_quantiles 组，各组内对远期收益取等权平均。
    返回 DataFrame：index=日期，columns=["Q1", ..., "Qn"]，Q1 为因子值最低组。
    某期某组无有效样本时该单元格为 NaN。
    """
    g = quantile_groups(factor, n_quantiles)
    f = _as_frame(forward_returns)
    idx = g.index.intersection(f.index)
    cols = g.columns.intersection(f.columns)
    g = g.loc[idx, cols]
    r = f.loc[idx, cols]
    mask = g.notna() & r.notna()

    out = {}
    for qi in range(1, n_quantiles + 1):
        member = mask & (g == qi)
        cnt = member.sum(axis=1)
        total = r.where(member).sum(axis=1)
        out[f"Q{qi}"] = total / cnt.replace(0, np.nan)
    return pd.DataFrame(out, index=idx)


def long_short_returns(factor: Panel, forward_returns: Panel,
                       n_quantiles: int = 5) -> pd.Series:
    """多空组合逐期收益：做多因子值最高分位组、做空最低分位组（等权、无成本）。"""
    qr = quantile_returns(factor, forward_returns, n_quantiles)
    return (qr.iloc[:, -1] - qr.iloc[:, 0]).rename("long_short")


def factor_coverage(factor: Panel) -> pd.Series:
    """逐截面因子覆盖率 = 非缺失资产数 / 总资产数。"""
    f = _as_frame(factor)
    n_assets = max(f.shape[1], 1)
    return (f.notna().sum(axis=1) / float(n_assets)).rename("coverage")


def factor_autocorrelation(factor: Panel, lag: int = 1,
                           min_obs: int = 2) -> pd.Series:
    """因子自相关：相邻期（滞后 lag）因子值的截面斯皮尔曼相关序列。

    衡量因子信号的持续性；前 lag 期无对比对象，为 NaN。
    """
    if lag < 1:
        raise ValueError("lag 必须 >= 1")
    f = _as_frame(factor)
    prev = f.shift(lag)
    fa, pa, mask = _align_pair(f, prev)
    return _row_corr(fa, pa, mask, min_obs, spearman=True).rename("autocorr")


def turnover(panel: Panel, kind: str = "weights") -> pd.Series:
    """换手率序列（单边口径：0.5 * sum |w_t - w_{t-1}|）。

    kind="weights":    panel 为目标权重面板（建议每行和为 1），NaN 视作 0 仓位；
    kind="membership": panel 为持仓名单面板（非零且非 NaN = 持有，如最高分位组的
                       0/1 指示），先转成组内等权权重再计算换手。

    首行无前一期参照，为 NaN。取值参考：0 = 完全不换仓，1 = 全部换血。
    """
    p = _as_frame(panel)
    if kind == "weights":
        w = p.fillna(0.0)
    elif kind == "membership":
        member = (p.notna() & (p != 0)).astype("float64")
        size = member.sum(axis=1)
        w = member.div(size.where(size > 0), axis=0).fillna(0.0)
    else:
        raise ValueError(f"未知换手类型: {kind!r}，仅支持 'weights' / 'membership'")
    # min_count=1：首行 diff 全为 NaN，换手应保持 NaN 而非 0
    return (w.diff().abs().sum(axis=1, min_count=1) * 0.5).rename("turnover")
