"""合成因子数据生成器，用于示例与测试（离线、固定 seed 可复现）。

make_factor_and_returns: 生成一个与远期收益具有可控截面相关性的因子面板；
make_style_exposures:    生成行业标签与对数市值面板，用于演示中性化。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

Panel = pd.DataFrame


def _asset_names(n_assets: int) -> list:
    return [f"S{i:03d}" for i in range(n_assets)]


def make_factor_and_returns(n_dates: int = 250, n_assets: int = 100,
                            true_ic: float = 0.05, seed: int = 42,
                            start: str = "2020-01-01", freq: str = "B",
                            return_vol: float = 0.02, persistence: float = 0.9,
                            market_drift: float = 0.0002,
                            market_vol: float = 0.01) -> Tuple[Panel, Panel]:
    """生成 (factor, forward_returns) 两个面板，index=交易日, columns=资产。

    构造方式
    --------
    1. 因子为逐资产 AR(1) 过程（persistence 控制时序持续性），再逐截面标准化，
       使每个截面上因子均值为 0、方差为 1；
    2. 远期收益 = return_vol * (true_ic * 因子 + sqrt(1 - true_ic^2) * 独立噪声)
       + 市场公共项。市场项对截面内所有资产相同，不影响截面相关性，
       因此每个截面上 corr(因子, 远期收益) 的总体值恰为 true_ic。

    对齐语义（防未来函数）：forward_returns 行 t = (t, t+1] 区间实现的收益，
    只依赖行 t 的因子值，可直接与因子面板同行配对计算 IC。
    """
    if n_dates < 1 or n_assets < 2:
        raise ValueError("需要 n_dates >= 1 且 n_assets >= 2")
    if not -1.0 <= true_ic <= 1.0:
        raise ValueError("true_ic 必须落在 [-1, 1]")

    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_dates, freq=freq)
    cols = _asset_names(n_assets)

    # 1) AR(1) 原始因子
    shocks = rng.standard_normal((n_dates, n_assets))
    raw = np.empty_like(shocks)
    raw[0] = shocks[0]
    for t in range(1, n_dates):
        raw[t] = persistence * raw[t - 1] + shocks[t]

    # 2) 逐截面标准化
    mu = raw.mean(axis=1, keepdims=True)
    sd = raw.std(axis=1, keepdims=True)
    z = (raw - mu) / np.where(sd > 0, sd, 1.0)

    # 3) 远期收益：信号 + 正交噪声 + 市场公共项
    noise = rng.standard_normal((n_dates, n_assets))
    idio = return_vol * (true_ic * z + np.sqrt(max(1.0 - true_ic ** 2, 0.0)) * noise)
    market = rng.normal(market_drift, market_vol, size=(n_dates, 1))
    fwd = idio + market

    factor = pd.DataFrame(z, index=idx, columns=cols)
    forward_returns = pd.DataFrame(fwd, index=idx, columns=cols)
    return factor, forward_returns


def make_style_exposures(n_dates: int = 250, n_assets: int = 100,
                         n_industries: int = 6, seed: int = 11,
                         start: str = "2020-01-01",
                         freq: str = "B") -> Tuple[Panel, Panel]:
    """生成 (industry_labels, log_mktcap) 风格暴露面板。

    - industry_labels: 行业标签面板（值为 "IND0".."INDk"），行业归属不随时间变化；
    - log_mktcap:      对数市值面板，由资产固定规模 + 小幅时序波动构成。

    两个面板与 make_factor_and_returns 使用相同 index/columns 约定，可直接
    传给 neutralize / FactorPipeline.standard。
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_dates, freq=freq)
    cols = _asset_names(n_assets)

    industry_of_asset = rng.integers(0, n_industries, size=n_assets)
    labels = pd.DataFrame(
        np.tile([f"IND{j}" for j in industry_of_asset], (n_dates, 1)),
        index=idx, columns=cols,
    )

    base_size = rng.normal(10.0, 1.0, size=(1, n_assets))
    wobble = rng.normal(0.0, 0.1, size=(n_dates, n_assets)).cumsum(axis=0) * 0.1
    log_mktcap = pd.DataFrame(base_size + wobble, index=idx, columns=cols)
    return labels, log_mktcap
