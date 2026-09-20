"""Kairos Factor —— 自研轻量因子研究库。

覆盖单因子研究的完整链路：
- preprocess: 去极值 / 标准化 / 排名归一 / 缺失填充 / 行业市值中性化（逐截面）；
- metrics:    IC / 秩 IC / IC 汇总 / 分层收益 / 多空收益 / 覆盖率 / 自相关 / 换手；
- evaluation: 多期限 IC 衰减与一站式 FactorReport；
- pipeline:   可复用的链式预处理流水线；
- sim:        可控 IC 的合成因子数据（离线、固定 seed）。

设计原则：逐截面变换、防未来函数（因子只与「其后实现」的远期收益配对）、
纯 numpy/pandas 依赖、可测试、可扩展。

数据约定：因子面板与远期收益面板均为 pandas.DataFrame，index=日期，columns=资产。
"""
from .evaluation import FactorReport, decay_analysis, forward_returns
from .metrics import (
    factor_autocorrelation,
    factor_coverage,
    ic,
    ic_summary,
    long_short_returns,
    quantile_groups,
    quantile_returns,
    rank_ic,
    turnover,
)
from .pipeline import FactorPipeline, PipelineStep
from .preprocess import (
    fill_cross_section,
    neutralize,
    rank_normalize,
    winsorize,
    zscore,
)
from .sim import make_factor_and_returns, make_style_exposures

__version__ = "0.1.0"

__all__ = [
    "winsorize", "zscore", "rank_normalize", "fill_cross_section", "neutralize",
    "ic", "rank_ic", "ic_summary", "quantile_groups", "quantile_returns",
    "long_short_returns", "factor_coverage", "factor_autocorrelation", "turnover",
    "decay_analysis", "forward_returns", "FactorReport",
    "FactorPipeline", "PipelineStep",
    "make_factor_and_returns", "make_style_exposures",
    "__version__",
]
