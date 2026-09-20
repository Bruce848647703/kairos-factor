"""Kairos Factor 演示：合成已知 IC 的因子 → 预处理流水线 → 完整评价报告。

运行： python examples/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_factor as kf
from kairos_factor import FactorPipeline, FactorReport

TRUE_IC = 0.08


def main():
    # ① 合成数据：因子与远期收益的截面相关性总体值 = TRUE_IC
    #    persistence=0：因子逐期独立、只预测下一期收益，便于观察 IC 随期限衰减
    factor, fwd = kf.make_factor_and_returns(
        n_dates=400, n_assets=120, true_ic=TRUE_IC, seed=42, persistence=0.0)
    labels, log_mktcap = kf.make_style_exposures(
        n_dates=400, n_assets=120, n_industries=6, seed=11)

    # 故意让因子被市值污染，用于演示中性化的效果
    contaminated = factor + 0.8 * kf.zscore(log_mktcap)
    size_corr_before = kf.rank_ic(contaminated, log_mktcap).mean()

    print("=" * 60)
    print(f"① 合成因子（真实 IC={TRUE_IC}）+ 市值污染（tilt=0.8）")
    print(f"   中性化前 因子×对数市值 平均秩相关: {size_corr_before:+.3f}")

    # ② 标准预处理流水线：去极值(MAD) → 行业市值中性化 → 标准化
    pipe = FactorPipeline.standard(industry_dummies=labels, log_mktcap=log_mktcap)
    print("=" * 60)
    print(f"② 预处理流水线: {pipe!r}")
    clean = pipe.fit_transform(contaminated)
    size_corr_after = kf.rank_ic(clean, log_mktcap).mean()
    print(f"   中性化后 因子×对数市值 平均秩相关: {size_corr_after:+.3f}")

    # ③ 一站式因子评价报告（期间收益 = 远期收益滞后一行，用于多期限 IC 衰减）
    period_returns = fwd.shift(1)
    report = FactorReport.build(
        clean, fwd, n_quantiles=5,
        prices_or_returns=period_returns, horizons=(1, 2, 3, 5, 10),
        kind="return", name="demo_factor")

    print("=" * 60)
    print("③ IC 汇总（报告标量指标）")
    print(report.summary_frame().to_string(index=False))

    print("=" * 60)
    print("④ 分层平均远期收益（Q1=因子最低组, Q5=最高组，均值应单调递增）")
    group_mean = report.quantile_returns.mean()
    print(group_mean.to_string())
    print(f"   多空(Q5-Q1)均值: {report.long_short.mean():+.5f} / 期，"
          f"IR: {report.summary['long_short_ir']:+.2f}")

    print("=" * 60)
    print("⑤ IC 衰减（信号只预测 1 期收益，期限越长预测力越弱）")
    print(report.decay.round(4).to_string())

    print("=" * 60)
    print("⑥ 持仓换手（最高分位组成员，等权持有）")
    groups = kf.quantile_groups(clean, n_quantiles=5)
    top = (groups == 5).astype(float)  # 最高分位组持仓指示（1=持有）
    to = kf.turnover(top, kind="membership")
    print(f"   平均单边换手率: {to.mean():.3f}")


if __name__ == "__main__":
    main()
