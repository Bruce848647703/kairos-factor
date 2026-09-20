"""因子综合评价模块。

- forward_returns: 由价格面板或期间收益面板推算任意期限的远期收益；
- decay_analysis:  多个远期期限上的 IC 衰减表；
- FactorReport:    一站式因子评价报告（dataclass），汇总 IC 序列、分层收益、
                   衰减、覆盖率与自相关，可导出 dict / 两列汇总表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .metrics import (
    factor_autocorrelation,
    factor_coverage,
    ic,
    ic_summary,
    long_short_returns,
    quantile_returns,
    rank_ic,
)

Panel = pd.DataFrame


def _as_frame(panel: Panel) -> Panel:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("输入必须是 pandas.DataFrame（index=日期, columns=资产）")
    return panel.astype("float64")


def _detect_kind(panel: Panel) -> str:
    """启发式判断面板是价格还是期间收益：全部为正且中位数明显大于 1 视作价格。

    净值型价格面板（起点 1.0 附近）可能被误判为收益，建议显式传 kind。
    """
    values = panel.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "return"
    return "price" if (finite.min() > 0.0 and np.median(finite) > 1.5) else "return"


def forward_returns(prices_or_returns: Panel, horizon: int = 1,
                    kind: str = "auto") -> Panel:
    """计算远期收益面板：行 t 的值 = 资产在 (t, t+horizon] 区间的收益。

    kind="price":  输入为价格面板，fwd_t = p_{t+h} / p_t - 1；
    kind="return": 输入为「期间收益」面板（行 s = (s-1, s] 收益），
                   fwd_t = prod(1 + r_{t+1..t+h}) - 1，窗口内含 NaN 则该格为 NaN；
    kind="auto":   用 _detect_kind 启发式判断。

    行对齐保持防未来函数语义：远期收益只使用 t 之后实现的数据，
    与因子面板行 t 直接配对即可计算 IC。
    """
    if horizon < 1:
        raise ValueError("horizon 必须 >= 1")
    p = _as_frame(prices_or_returns)
    k = _detect_kind(p) if kind == "auto" else kind
    if k == "price":
        return p.shift(-horizon) / p - 1.0
    if k == "return":
        growth = 1.0 + p
        return growth.rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    raise ValueError(f"未知面板类型: {kind!r}，仅支持 'price' / 'return' / 'auto'")


def decay_analysis(factor: Panel, prices_or_returns: Panel,
                   horizons: Iterable[int] = (1, 2, 3, 5, 10),
                   kind: str = "auto") -> pd.DataFrame:
    """IC 衰减分析：同一因子在多个远期期限上的预测能力对比。

    对每个 horizon 由 prices_or_returns 推算远期收益，分别计算皮尔逊 IC 与
    秩 IC 序列并汇总。返回 DataFrame：index=horizon，
    columns=[ic_mean, rank_ic_mean, ic_ir, rank_ic_ir,
             ic_positive_ratio, rank_ic_positive_ratio, count]。
    """
    rows: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        fwd = forward_returns(prices_or_returns, horizon=h, kind=kind)
        s_pearson = ic_summary(ic(factor, fwd))
        s_spearman = ic_summary(rank_ic(factor, fwd))
        rows[int(h)] = {
            "ic_mean": s_pearson["mean"],
            "rank_ic_mean": s_spearman["mean"],
            "ic_ir": s_pearson["ir"],
            "rank_ic_ir": s_spearman["ir"],
            "ic_positive_ratio": s_pearson["positive_ratio"],
            "rank_ic_positive_ratio": s_spearman["positive_ratio"],
            "count": float(s_pearson["count"]),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "horizon"
    return out.sort_index()


@dataclass
class FactorReport:
    """因子评价报告：汇总一次因子分析的全部核心产物。

    字段
    ----
    ic / rank_ic:          逐期 IC 序列（与远期收益同频）；
    quantile_returns:      分层组合逐期收益（Q1=因子最低组）；
    long_short:            多空（Qn - Q1）逐期收益；
    coverage:              逐期因子覆盖率；
    autocorrelation:       逐期因子自相关（滞后 1 期秩相关）；
    decay:                 IC 衰减表（未提供价格/收益面板时为 None）；
    summary:               标量汇总字典（ic_*/rank_ic_*/long_short_mean 等）。
    """

    ic: pd.Series
    rank_ic: pd.Series
    quantile_returns: Panel
    long_short: pd.Series
    coverage: pd.Series
    autocorrelation: pd.Series
    name: str = "factor"
    n_quantiles: int = 5
    n_periods: int = 0
    n_assets: int = 0
    decay: Optional[pd.DataFrame] = None
    summary: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, factor: Panel, fwd_returns: Panel, n_quantiles: int = 5,
              prices_or_returns: Optional[Panel] = None,
              horizons: Iterable[int] = (1, 2, 3, 5, 10),
              kind: str = "auto", name: str = "factor") -> "FactorReport":
        """由因子面板与（1 期）远期收益面板构建完整报告。

        prices_or_returns 非空时额外计算 IC 衰减表；kind 传给 decay_analysis。
        """
        f = _as_frame(factor)
        ic_s = ic(f, fwd_returns)
        rank_s = rank_ic(f, fwd_returns)
        qr = quantile_returns(f, fwd_returns, n_quantiles)
        ls = long_short_returns(f, fwd_returns, n_quantiles)
        cov = factor_coverage(f)
        autocorr = factor_autocorrelation(f)
        decay = None
        if prices_or_returns is not None:
            decay = decay_analysis(f, prices_or_returns, horizons=horizons, kind=kind)

        summary: Dict[str, float] = {}
        for prefix, series in (("ic", ic_s), ("rank_ic", rank_s)):
            for key, value in ic_summary(series).items():
                summary[f"{prefix}_{key}"] = value
        summary["long_short_mean"] = float(ls.mean()) if ls.notna().any() else float("nan")
        summary["long_short_ir"] = (
            float(ls.mean() / ls.std(ddof=1))
            if ls.notna().sum() > 1 and float(ls.std(ddof=1)) > 1e-14 else 0.0
        )
        summary["coverage_mean"] = float(cov.mean()) if cov.notna().any() else float("nan")
        summary["autocorr_mean"] = (
            float(autocorr.mean()) if autocorr.notna().any() else float("nan")
        )

        return cls(
            ic=ic_s, rank_ic=rank_s, quantile_returns=qr, long_short=ls,
            coverage=cov, autocorrelation=autocorr, name=name,
            n_quantiles=n_quantiles, n_periods=int(f.shape[0]),
            n_assets=int(f.shape[1]), decay=decay, summary=summary,
        )

    def to_dict(self) -> Dict[str, object]:
        """导出标量汇总字典（summary + 元信息），便于日志与落库。"""
        out: Dict[str, object] = {
            "name": self.name,
            "n_periods": self.n_periods,
            "n_assets": self.n_assets,
            "n_quantiles": self.n_quantiles,
        }
        out.update(self.summary)
        return out

    def summary_frame(self) -> pd.DataFrame:
        """把 to_dict 整理成两列（metric/value）DataFrame，便于打印。"""
        d = self.to_dict()
        return pd.DataFrame({"metric": list(d.keys()), "value": list(d.values())})
