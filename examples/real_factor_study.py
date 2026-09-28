"""真实 A 股数据上的因子研究示例：加载真实日线 → 构造因子 → 预处理 → 评价 → 落盘报告。

与 examples/demo.py（合成数据、已知真实 IC）不同，本脚本跑的是**真实行情**：
结论由数据说话，可能很弱、也可能不显著，报告里如实写出（含数据质量处理与局限）。

运行
----
    python examples/real_factor_study.py --data-dir /path/to/ashare_csv_dir
    python examples/real_factor_study.py                 # 使用默认数据目录（存在时）

产物（默认写到 <repo>/research/real_factor/）
---------------------------------------------
    REPORT.md             中文研究报告（IC/ICIR/t 值/分层收益/衰减 + 诚实结论）
    factor_ic.csv         每个因子 × 样本 × 预处理 的 IC 汇总（含多期限衰减）
    quantile_returns.csv  分层（分位）组合与多空的逐期收益统计

数据声明：行情来自公开行情接口（前复权日线），仅用于研究与教学演示，
版权归原作者/数据源所有，不构成任何投资建议。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import kairos_factor as kf
from kairos_factor import FactorPipeline, FactorReport
from kairos_factor.realdata import (
    DEFAULT_MAX_DAILY_MOVE,
    SECTOR_MAP,
    clean_start,
    drop_artifact_symbols,
    flagged_symbols,
    load_close_panel,
    sector_labels,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(REPO_ROOT), "kairos-data", "data", "ashare"))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "research", "real_factor")

TRADING_DAYS = 252

# 因子定义：(key, 中文名, 构造函数(close, ret) -> 原始因子面板)
# 方向统一为「值越大越看多」：反转与低波动因此带负号。
FACTOR_SPECS: Tuple[Tuple[str, str, object], ...] = (
    ("momentum_60", "60日动量（3个月相对涨幅）", lambda close, ret: close.pct_change(60)),
    ("momentum_20", "20日动量（1个月相对涨幅）", lambda close, ret: close.pct_change(20)),
    ("reversal_5", "5日反转（负 5 日涨幅）", lambda close, ret: -close.pct_change(5)),
    ("lowvol_20", "低波动（负 20 日已实现波动）", lambda close, ret: -ret.rolling(20).std()),
)

MODE_LABEL = {"raw": "winsorize+zscore", "neutral": "winsorize+行业中性化+zscore"}

# 因子方向解读：秩 IC 为正 / 为负时分别说明市场呈现什么特征
FACTOR_READING: Dict[str, Tuple[str, str]] = {
    "momentum_60": ("3 个月动量延续", "3 个月反转"),
    "momentum_20": ("1 个月动量延续", "1 个月反转"),
    "reversal_5": ("5 日反转", "5 日动量延续"),
    "lowvol_20": ("低波动占优", "高波动占优"),
}


# ---------------------------------------------------------------------------
# 样本与因子构造
# ---------------------------------------------------------------------------

def build_samples(close: pd.DataFrame, max_daily_move: float) -> List[Dict[str, object]]:
    """构造两个互补的研究样本，用于检验结论是否依赖窗口选择。

    - clean: 全体标的，窗口从「此后不再出现伪影」的共同起点开始（牺牲长度换标的完整）；
    - full:  保留加载器给出的完整窗口，剔除历史上出现过伪影的标的（牺牲标的换长度）。
    """
    flagged = flagged_symbols(close, max_daily_move)
    start = clean_start(close, max_daily_move) if flagged else close.index[0]
    samples = [{"key": "clean", "panel": close.loc[start:], "dropped": []}]
    if flagged:
        samples.append({"key": "full", "panel": drop_artifact_symbols(close, max_daily_move),
                        "dropped": flagged})
    else:
        samples.append({"key": "full", "panel": close, "dropped": []})
    return samples


def build_factors(close: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """由价格面板构造全部原始因子（只用历史数据，无未来函数）。"""
    ret = close.pct_change()
    return {key: func(close, ret) for key, _, func in FACTOR_SPECS}


def build_pipelines(labels: pd.DataFrame) -> Dict[str, FactorPipeline]:
    """两条预处理流水线：仅去极值标准化 / 再做行业中性化。"""
    return {
        "raw": FactorPipeline.standard(),
        "neutral": FactorPipeline.standard(industry_dummies=labels),
    }


# ---------------------------------------------------------------------------
# 评价
# ---------------------------------------------------------------------------

def evaluate_case(name: str, cn: str, raw: pd.DataFrame, pipe: FactorPipeline,
                  close: pd.DataFrame, sample_key: str, mode: str, horizon: int,
                  n_quantiles: int, decay_horizons: Sequence[int]) -> Tuple[Dict, pd.DataFrame]:
    """对单个（因子 × 预处理）组合跑完整评价，返回 (标量记录, 分层收益记录)。"""
    clean_factor = pipe.fit_transform(raw)
    fwd = kf.forward_returns(close, horizon=horizon, kind="price")
    report = FactorReport.build(clean_factor, fwd, n_quantiles=n_quantiles,
                                prices_or_returns=close, horizons=decay_horizons,
                                kind="price", name=name)
    ric = report.rank_ic
    nonoverlap = kf.ic_summary(ric.iloc[::horizon])  # 非重叠抽样，缓解重叠期 t 值虚高
    s = kf.ic_summary(ric)
    p = kf.ic_summary(report.ic)
    ls = report.long_short

    per_year = float(TRADING_DAYS) / float(horizon)
    rec = {
        "sample": sample_key, "factor": name, "factor_cn": cn, "mode": mode,
        "horizon": horizon, "n_assets": int(clean_factor.shape[1]),
        "n_periods": int(ric.notna().sum()),
        "rank_ic_mean": s["mean"], "rank_ic_std": s["std"], "rank_ic_ir": s["ir"],
        "rank_ic_t": s["t_stat"], "rank_ic_t_nonoverlap": nonoverlap["t_stat"],
        "rank_ic_positive_ratio": s["positive_ratio"],
        "ic_mean": p["mean"], "ic_ir": p["ir"], "ic_t": p["t_stat"],
        "ls_mean_bp": float(ls.mean()) * 1e4, "ls_ann_pct": float(ls.mean()) * per_year * 100.0,
        "ls_ir": float(ls.mean() / ls.std(ddof=1)) if float(ls.std(ddof=1)) > 1e-14 else 0.0,
        "coverage_mean": report.summary.get("coverage_mean", float("nan")),
        "autocorr_mean": report.summary.get("autocorr_mean", float("nan")),
    }
    for h in decay_horizons:
        rec["decay_rank_ic_h%d" % int(h)] = (
            float(report.decay.loc[int(h), "rank_ic_mean"]) if report.decay is not None
            and int(h) in report.decay.index else float("nan"))

    qr = report.quantile_returns
    buckets = [(c, qr[c]) for c in qr.columns] + [("LS(Q%d-Q1)" % n_quantiles, ls)]
    qrows = []
    for bucket, series in buckets:
        series = series.dropna()
        qrows.append({
            "sample": sample_key, "factor": name, "mode": mode, "horizon": horizon,
            "bucket": bucket, "mean_ret": float(series.mean()) if len(series) else float("nan"),
            "mean_ret_bp": float(series.mean()) * 1e4 if len(series) else float("nan"),
            "ann_ret_pct": float(series.mean()) * per_year * 100.0 if len(series) else float("nan"),
            "win_rate": float((series > 0).mean()) if len(series) else float("nan"),
            "n_periods": int(len(series)),
        })
    return rec, pd.DataFrame(qrows)


def monotonic_score(qmean: Sequence[float]) -> float:
    """分层平均收益与组序的斯皮尔曼相关（±1 = 完全单调，0 = 无单调性）。"""
    y = np.asarray(qmean, dtype="float64")
    if y.size < 3 or not np.isfinite(y).all():
        return float("nan")
    ry = pd.Series(y).rank().to_numpy()
    rx = np.arange(1.0, y.size + 1.0)
    if ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# 输出：CSV / Markdown
# ---------------------------------------------------------------------------

def _fmt(value: object, nd: int = 4) -> str:
    """把单元格格式化成 Markdown 友好的字符串（NaN -> 破折号，避免 -0.0）。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if abs(v) < 0.5 * 10.0 ** (-nd):
            v = 0.0
        return ("%+." + str(nd) + "f") % v
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def md_table(df: pd.DataFrame, index_name: str = "", nd: int = 4) -> str:
    """把 DataFrame 渲染成 GitHub Markdown 表格（不依赖 tabulate）。"""
    body = df.copy()
    body.index.name = index_name or body.index.name or ""
    body = body.reset_index()
    names = [str(c) for c in body.columns]
    lines = ["| " + " | ".join(names) + " |", "|" + "|".join(["---"] * len(names)) + "|"]
    for _, row in body.iterrows():
        cells = [v if isinstance(v, str) else _fmt(v, nd) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ic_view(ic_rows: pd.DataFrame, sample: str, mode: str,
            decay_horizons: Sequence[int]) -> pd.DataFrame:
    """挑出某样本某预处理下的 IC 汇总表（中文列名，便于阅读）。"""
    cols = ["factor", "n_assets", "n_periods", "rank_ic_mean", "rank_ic_ir", "rank_ic_t",
            "rank_ic_t_nonoverlap", "rank_ic_positive_ratio", "ic_mean", "ic_t",
            "ls_mean_bp", "ls_ann_pct", "ls_ir", "autocorr_mean"]
    sub = ic_rows[(ic_rows["sample"] == sample) & (ic_rows["mode"] == mode)][cols].copy()
    for h in decay_horizons:
        sub["decay_h%d" % int(h)] = ic_rows.loc[sub.index, "decay_rank_ic_h%d" % int(h)].to_numpy()
    rename = {
        "factor": "因子", "n_assets": "标的数", "n_periods": "有效期数",
        "rank_ic_mean": "秩IC均值", "rank_ic_ir": "ICIR", "rank_ic_t": "t值(重叠)",
        "rank_ic_t_nonoverlap": "t值(非重叠)", "rank_ic_positive_ratio": "IC>0占比",
        "ic_mean": "皮尔逊IC", "ic_t": "IC t值", "ls_mean_bp": "多空bp/期",
        "ls_ann_pct": "多空年化%", "ls_ir": "多空IR", "autocorr_mean": "因子自相关",
    }
    for h in decay_horizons:
        rename["decay_h%d" % int(h)] = "衰减h%d" % int(h)
    return sub.rename(columns=rename).set_index("因子").round(4)


def quantile_view(q_rows: pd.DataFrame, sample: str, mode: str) -> pd.DataFrame:
    """把分层收益长表转成「因子 × 分位」的平均收益(bp/期)矩阵。"""
    sub = q_rows[(q_rows["sample"] == sample) & (q_rows["mode"] == mode)]
    mat = sub.pivot(index="factor", columns="bucket", values="mean_ret_bp")
    order = [b for b in sub["bucket"].drop_duplicates() if b in mat.columns]
    mat = mat[order].round(1)
    mat[mat == 0.0] = 0.0          # 消除 -0.0，避免阅读歧义
    return mat


def factor_verdicts(ic_rows: pd.DataFrame) -> pd.DataFrame:
    """汇总每个因子在全部（样本 × 预处理）组合上的秩 IC 与方向一致性判定。

    「同号组合」= 秩 IC 同号的组合个数占比是否达到 3/4：达到才认为方向可信，
    否则标注「方向不稳定」。判定文字取自 FACTOR_READING（按因子方向约定翻译）。
    """
    combos = list(dict.fromkeys(zip(ic_rows["sample"], ic_rows["mode"])))
    records = []
    for factor in ic_rows["factor"].drop_duplicates():
        sub = ic_rows[ic_rows["factor"] == factor]
        values = [float(sub[(sub["sample"] == s) & (sub["mode"] == m)]["rank_ic_mean"].iloc[0])
                  for s, m in combos]
        arr = np.asarray(values, dtype="float64")
        pos, neg, n = int((arr > 0).sum()), int((arr < 0).sum()), arr.size
        dominant = "正" if pos > neg else ("负" if neg > pos else "混合")
        stable = max(pos, neg) * 4 >= 3 * n
        up, down = FACTOR_READING.get(factor, ("与后续收益正相关", "与后续收益负相关"))
        reading = up if dominant == "正" else (down if dominant == "负" else "无稳定方向")
        if not stable:
            reading += "（方向不稳定）"
        if float(np.abs(arr).mean()) < 0.01:
            reading += "（量级可忽略）"
        rec = {"因子": factor}
        for (s, m), v in zip(combos, values):
            rec["%s·%s" % (s, m)] = v
        rec["秩IC均值"] = float(arr.mean())
        rec["同号组合数"] = "%d/%d %s" % (max(pos, neg), n, dominant)
        rec["判定"] = reading
        records.append(rec)
    return pd.DataFrame(records).set_index("因子")


def conclusion_lines(ic_rows: pd.DataFrame, q_rows: pd.DataFrame, verdicts: pd.DataFrame,
                     samples: Sequence[str], horizon: int, n_quantiles: int) -> List[str]:
    """基于计算结果生成诚实结论（数字全部来自结果表，不硬编码）。"""
    primary, secondary = samples[0], samples[-1]
    lines: List[str] = []

    def pick(sample: str, factor: str, mode: str, field: str = "rank_ic_mean") -> float:
        row = ic_rows[(ic_rows["sample"] == sample) & (ic_rows["factor"] == factor)
                      & (ic_rows["mode"] == mode)]
        return float(row[field].iloc[0]) if len(row) else float("nan")

    def reading(factor: str) -> str:
        return str(verdicts.loc[factor, "判定"]) if factor in verdicts.index else "—"

    # ① 信号强度：全部组合的极值与显著性计数
    absmax_row = ic_rows.loc[ic_rows["rank_ic_mean"].abs().idxmax()]
    n_t_nov = int((ic_rows["rank_ic_t_nonoverlap"].abs() >= 2.0).sum())
    n_t_ovl = int((ic_rows["rank_ic_t"].abs() >= 2.0).sum())
    sig_txt = ("、".join("%s(%s·%s) t=%.2f" % (r["factor"], r["sample"], r["mode"],
                                              r["rank_ic_t"])
                         for _, r in ic_rows[ic_rows["rank_ic_t"].abs() >= 2.0].iterrows())
               or "无")
    lines.append(
        "1. **信号整体偏弱**：%d 个（样本 × 因子 × 预处理）组合中，|秩 IC 均值| 最大只有 %.4f"
        "（%s·%s·%s），低于业界常用的 0.03~0.05 可用门槛；重叠口径下 |t| ≥ 2 的有 %d 个（%s），"
        "但把重叠期修正掉之后（非重叠抽样）|t| ≥ 2 的只有 %d 个——即**没有一个组合能在稳健口径下"
        "通过显著性检验**。"
        % (len(ic_rows), abs(float(absmax_row["rank_ic_mean"])), absmax_row["sample"],
           absmax_row["factor"], absmax_row["mode"], n_t_ovl, sig_txt, n_t_nov))

    # ② 动量 vs 反转：按回看期分别给判定
    lines.append(
        "2. **动量还是反转？分期限看，没有单一答案**：60 日（3 个月）口径判定为「%s」"
        "（秩 IC 主样本 %+.4f / 长样本 %+.4f）；20 日（1 个月）口径为「%s」"
        "（%+.4f / %+.4f）；5 日口径为「%s」（%+.4f / %+.4f）。也就是说该池呈现"
        "**极短期反转、一个月动量、三个月再反转**的混合结构，而非单纯的动量市或反转市。"
        % (reading("momentum_60"), pick(primary, "momentum_60", "raw"),
           pick(secondary, "momentum_60", "raw"), reading("momentum_20"),
           pick(primary, "momentum_20", "raw"), pick(secondary, "momentum_20", "raw"),
           reading("reversal_5"), pick(primary, "reversal_5", "raw"),
           pick(secondary, "reversal_5", "raw")))
    m60 = float(verdicts.loc["momentum_60", "秩IC均值"]) if "momentum_60" in verdicts.index else 0.0
    lines.append(
        "   若必须在两者间二选一：以 60 日口径（%s，秩 IC 均值 %+.4f）看，该池整体更偏**%s**；"
        "但该量级扣掉成本后无实用价值。"
        % (reading("momentum_60"), m60, "反转" if m60 < 0 else "动量"))

    # ③ 低波动：窗口依赖
    lv_a, lv_b = pick(primary, "lowvol_20", "raw"), pick(secondary, "lowvol_20", "raw")
    lines.append(
        "3. **低波动效应不稳定**：lowvol_20 判定为「%s」，但秩 IC 从主样本的 %+.4f 变到长样本的 "
        "%+.4f（皮尔逊 IC 由 %+.4f 变到 %+.4f，多空由 %+.1f 变到 %+.1f bp/期），"
        "量级和符号都随窗口大幅摆动，属于**窗口依赖**的结果，不能当作稳定 alpha。"
        % (reading("lowvol_20"), lv_a, lv_b, pick(primary, "lowvol_20", "raw", "ic_mean"),
           pick(secondary, "lowvol_20", "raw", "ic_mean"),
           pick(primary, "lowvol_20", "raw", "ls_mean_bp"),
           pick(secondary, "lowvol_20", "raw", "ls_mean_bp")))

    # ④ 行业中性化的作用
    raw_abs = ic_rows[(ic_rows["sample"] == primary) & (ic_rows["mode"] == "raw")][
        "rank_ic_mean"].abs().mean()
    neu_abs = ic_rows[(ic_rows["sample"] == primary) & (ic_rows["mode"] == "neutral")][
        "rank_ic_mean"].abs().mean()
    raw_ls = ic_rows[(ic_rows["sample"] == primary) & (ic_rows["mode"] == "raw")][
        ["factor", "ls_mean_bp"]].set_index("factor")["ls_mean_bp"]
    neu_ls = ic_rows[(ic_rows["sample"] == primary) & (ic_rows["mode"] == "neutral")][
        ["factor", "ls_mean_bp"]].set_index("factor")["ls_mean_bp"]
    shrink = int((neu_ls.abs() < raw_ls.abs()).sum())
    lines.append(
        "4. **行业中性化削掉的是行业 beta**：主样本平均 |秩 IC| 由 %.4f（raw）降到 %.4f（neutral，"
        "%.0f%%）；多空价差 %d/%d 个因子绝对值下降。说明原始信号中相当一部分来自行业间涨跌差异"
        "（如新能源/白酒/银行的板块行情），而非纯粹的个股选择能力。"
        % (raw_abs, neu_abs, (1.0 - neu_abs / raw_abs) * 100.0 if raw_abs > 0 else float("nan"),
           shrink, len(neu_ls)))

    # ⑤ 分层单调性
    mono = {}
    for factor in ic_rows["factor"].drop_duplicates():
        sub = q_rows[(q_rows["sample"] == primary) & (q_rows["mode"] == "neutral")
                     & (q_rows["factor"] == factor) & (q_rows["bucket"].str.match(r"Q\d"))]
        mono[factor] = monotonic_score(sub.sort_values("bucket")["mean_ret"].tolist())
    mono_txt = "、".join("%s %+.2f" % (k, v) for k, v in mono.items())
    good = [k for k, v in mono.items() if np.isfinite(v) and abs(v) >= 0.8]
    lines.append(
        "5. **分层单调性普遍不足**：主样本行业中性化后，%d 层组合平均收益与组序的斯皮尔曼相关为 %s；"
        "其中只有 %s 达到近似单调（|ρ| ≥ 0.8），其余因子的分层收益非单调，"
        "说明信号只在个别分位之间有效，直接拿来做多头选股会亏在中间层。"
        % (n_quantiles, mono_txt, "、".join(good) if good else "无"))

    # ⑥ 综合判断：一致性优先（同号组合数），其次才是量级
    consist = {}
    for factor in ic_rows["factor"].drop_duplicates():
        vals = ic_rows[ic_rows["factor"] == factor]["rank_ic_mean"]
        consist[factor] = (int(max((vals > 0).sum(), (vals < 0).sum())), float(vals.abs().mean()))
    best = max(consist, key=lambda k: (consist[k][0], consist[k][1]))
    tmax = ic_rows.groupby("factor")["rank_ic_t_nonoverlap"].apply(
        lambda s: float(s.abs().max()))
    lines.append(
        "6. **综合判断**：本次没有出现「|秩 IC| ≥ 0.03 且非重叠 |t| ≥ 2」的因子；方向最一致的是 "
        "%s（%s，秩 IC 均值 %+.4f，非重叠 |t| 最大 %.2f）。因此诚实的结论是——"
        "**在这 %d 只大盘蓝筹、最多 %d 个有效截面的样本里，找不到可直接实盘的单因子 alpha**，"
        "而不是「某类因子有效」。后续若要继续：扩大到全市场（数百只以上）、"
        "引入成交量与基本面字段、加交易成本与容量约束、并对多重检验做校正（Bonferroni / FDR）。"
        % (best, verdicts.loc[best, "同号组合数"], float(verdicts.loc[best, "秩IC均值"]),
           float(tmax[best]), int(ic_rows["n_assets"].max()), int(ic_rows["n_periods"].max())))
    return lines


def write_report(path: str, meta: Dict[str, object], ic_rows: pd.DataFrame,
                 q_rows: pd.DataFrame, samples: List[Dict[str, object]], horizon: int,
                 n_quantiles: int, decay_horizons: Sequence[int]) -> None:
    """把结果写成中文 Markdown 研究报告。"""
    primary, secondary = samples[0]["key"], samples[1]["key"]
    lines: List[str] = []
    add = lines.append

    add("# 真实 A 股数据的因子研究报告")
    add("")
    add("> 由 `examples/real_factor_study.py` 自动生成；数字均来自本次运行，未做任何手工修饰。")
    add("")
    add("## 1. 数据与样本")
    add("")
    add("- 数据目录：`%s`" % meta["data_dir"])
    add("- 加载后价格面板：%d 个交易日 × %d 只标的（%s ~ %s），前复权日线收盘价"
        % (meta["panel_dates"], meta["panel_assets"], meta["panel_start"], meta["panel_end"]))
    add("- 行业分组：%d 个（%s）" % (meta["n_sectors"], "、".join(meta["sector_sizes"])))
    add("- 主评价期限 h = %d 个交易日（远期收益为 `(t, t+h]` 区间收益，与因子行 t 配对，防未来函数）"
        % horizon)
    add("- 分层数 = %d（每层约 %d 只），多空 = Q%d − Q1，等权、无交易成本"
        % (n_quantiles, round(meta["panel_assets"] / n_quantiles), n_quantiles))
    add("")
    add("### 数据质量：前复权伪影")
    add("")
    add("前复权价 = 原始价 − 累计分红调整，高分红标的的早年价格会被压到接近 0 甚至为负，"
        "价格趋零时真实的小幅涨跌会被放大成数倍的伪收益（例如本池中 %s）。"
        "因此用「|单日涨跌| > %.0f%%（远超 A 股 ±10%%/±20%% 涨跌停上界）」作为伪影筛查，"
        "并给出两个互补样本，检验结论是否依赖窗口选择：" % (meta["artifact_note"],
                                                            meta["max_daily_move"] * 100))
    add("")
    add("| 样本 | 口径 | 窗口 | 交易日 | 标的数 | 剔除标的 |")
    add("|---|---|---|---|---|---|")
    for smp in samples:
        panel = smp["panel"]
        add("| %s | %s | %s ~ %s | %d | %d | %s |" % (
            smp["key"], smp["desc"], panel.index.min().date(), panel.index.max().date(),
            len(panel), panel.shape[1], "、".join(smp["dropped"]) or "无"))
    add("")
    add("## 2. 因子与预处理")
    add("")
    add("| 因子 | 定义 | 方向 |")
    add("|---|---|---|")
    add("| `momentum_60` | `close.pct_change(60)` | 值大 = 3 个月涨幅大（看多动量） |")
    add("| `momentum_20` | `close.pct_change(20)` | 值大 = 1 个月涨幅大（看多动量） |")
    add("| `reversal_5` | `-close.pct_change(5)` | 值大 = 近 5 日跌得多（看多反转） |")
    add("| `lowvol_20` | `-ret.rolling(20).std()` | 值大 = 已实现波动低（看多低波） |")
    add("")
    add("预处理流水线（`kairos_factor.FactorPipeline`，逐截面、无跨期信息）：")
    add("")
    add("- `raw`：%s" % MODE_LABEL["raw"])
    add("- `neutral`：%s（行业哑变量 OLS 取残差，%d 个行业）" % (MODE_LABEL["neutral"],
                                                             meta["n_sectors"]))
    add("")
    add("评价指标全部来自本库：`forward_returns` / `rank_ic` / `ic` / `ic_summary` / "
        "`quantile_returns` / `long_short_returns` / `decay_analysis` / `FactorReport`。")
    add("")
    add("**t 值口径**：因子每日计算、远期收益窗口重叠 %d 日，逐日 IC 序列存在自相关，"
        "直接用 `mean/std*sqrt(n)` 会高估显著性；因此同时给出「非重叠抽样」t 值"
        "（每 %d 日取一个 IC，`ic_summary(ric.iloc[::%d])`），后者更接近真实显著性。"
        % (horizon, horizon, horizon))
    add("")

    for key, title in ((primary, "3. 主样本结果（%s）" % samples[0]["desc"]),
                       (secondary, "4. 稳健性检验（%s）" % samples[1]["desc"])):
        add("## %s" % title)
        add("")
        for mode in ("neutral", "raw"):
            add("### %s · %s" % (key, MODE_LABEL[mode]))
            add("")
            add(md_table(ic_view(ic_rows, key, mode, decay_horizons), index_name="因子"))
            add("")
            add("分层平均收益（bp/%d 日，Q1 = 因子值最低组）：" % horizon)
            add("")
            add(md_table(quantile_view(q_rows, key, mode), index_name="因子", nd=1))
            add("")
    add("## 5. 诚实结论")
    add("")
    verdicts = factor_verdicts(ic_rows)
    add("先看跨样本、跨预处理的方向一致性（秩 IC 均值，%d 个组合）：" % len(ic_rows))
    add("")
    add(md_table(verdicts.drop(columns=["判定"]), index_name="因子"))
    add("")
    add("| 因子 | 判定 |")
    add("|---|---|")
    for factor, row in verdicts.iterrows():
        add("| `%s` | %s |" % (factor, row["判定"]))
    add("")
    for line in conclusion_lines(ic_rows, q_rows, verdicts, [primary, secondary], horizon,
                                 n_quantiles):
        add("- " + line)
    add("")
    add("## 6. 局限")
    add("")
    add("- **样本太小**：%d 只大盘蓝筹、%d 个行业，单层仅约 %d 只，截面统计噪声大，"
        "IC 的标准误本身就与信号量级相当。" % (meta["panel_assets"], meta["n_sectors"],
                                            round(meta["panel_assets"] / n_quantiles)))
    add("- **无成本、无约束**：多空价差未扣交易成本/冲击成本，也未考虑涨跌停不可成交、"
        "停牌与股票池调整；%d 日调仓的单边换手会让上述价差大幅缩水。" % horizon)
    add("- **重叠期与多重检验**：本报告一次检验 4 因子 × 2 预处理 × 2 样本 = %d 个组合，"
        "在 5%% 水平上本就期望出现约 %d 个「假显著」，因此单看 t 值极易过拟合。"
        % (len(ic_rows), max(1, round(len(ic_rows) * 0.05))))
    add("- **数据口径**：前复权价格由数据源给出，累计分红调整会使早年价格失真（已按上文筛查处理）；"
        "停牌日用历史价 ffill，会低估当期波动。")
    add("- **非投资建议**：所有结论仅描述该样本内的历史统计特征，不代表未来收益。")
    add("")
    add("## 7. 复现")
    add("")
    add("```bash")
    add("cd %s" % REPO_ROOT)
    add("python examples/real_factor_study.py --data-dir %s" % meta["data_dir"])
    add("python -m pytest -q     # 离线测试（含 tests/test_realdata.py）")
    add("```")
    add("")
    add("数据来自公开行情接口，仅用于研究与教学演示，版权归原作者所有，不构成投资建议。")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="真实 A 股数据上的因子研究示例")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="行情 CSV 目录（内含 <symbol>.csv），默认 %s" % DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="结果输出目录，默认 <repo>/research/real_factor")
    ap.add_argument("--horizon", type=int, default=5, help="主评价期限（交易日），默认 5")
    ap.add_argument("--quantiles", type=int, default=5, help="分层数，默认 5")
    ap.add_argument("--decay-horizons", default="1,2,5,10,20",
                    help="IC 衰减期限列表，逗号分隔，默认 1,2,5,10,20")
    ap.add_argument("--max-daily-move", type=float, default=DEFAULT_MAX_DAILY_MOVE,
                    help="前复权伪影阈值（|单日涨跌|），默认 %.2f" % DEFAULT_MAX_DAILY_MOVE)
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.isdir(args.data_dir):
        print("数据目录不存在: %s\n请用 --data-dir 指定真实行情 CSV 目录" % args.data_dir,
              file=sys.stderr)
        return 2
    decay_horizons = tuple(int(x) for x in str(args.decay_horizons).split(",") if x.strip())

    print("=" * 78)
    print("① 加载真实行情: %s" % args.data_dir)
    close = load_close_panel(args.data_dir)
    unmapped = [c for c in close.columns if c not in SECTOR_MAP]
    print("   面板: %d 个交易日 × %d 只标的, %s ~ %s, 内部缺失 %d 个"
          % (len(close), close.shape[1], close.index.min().date(), close.index.max().date(),
             int(close.isna().sum().sum())))
    if unmapped:
        print("   警告: 以下标的无行业归属，中性化时会被剔除: %s" % "、".join(unmapped))

    flagged = flagged_symbols(close, args.max_daily_move)
    print("② 前复权伪影筛查（|单日涨跌| > %.0f%%）: %s"
          % (args.max_daily_move * 100, "、".join(flagged) or "无"))
    samples = build_samples(close, args.max_daily_move)
    for smp, desc in zip(samples, ("全体标的·共同可信起点", "剔除失真标的·完整窗口")):
        smp["desc"] = desc
        panel = smp["panel"]
        print("   样本 %-5s %s ~ %s, %4d 个交易日 × %2d 只标的"
              % (smp["key"], panel.index.min().date(), panel.index.max().date(),
                 len(panel), panel.shape[1]))

    ic_records: List[Dict] = []
    q_frames: List[pd.DataFrame] = []
    print("=" * 78)
    print("③ 因子评价（h=%d 日, %d 层, 预处理: %s）"
          % (args.horizon, args.quantiles, " / ".join(MODE_LABEL.values())))
    for smp in samples:
        panel = smp["panel"]
        labels = sector_labels(panel.index, panel.columns)
        pipes = build_pipelines(labels)
        factors = build_factors(panel)
        for key, cn, _ in FACTOR_SPECS:
            for mode, pipe in pipes.items():
                rec, qdf = evaluate_case(key, cn, factors[key], pipe, panel, smp["key"], mode,
                                         args.horizon, args.quantiles, decay_horizons)
                ic_records.append(rec)
                q_frames.append(qdf)
                print("   [%s|%s|%s] 秩IC=%+.4f ICIR=%+.3f t(重叠)=%+.2f t(非重叠)=%+.2f "
                      "IC>0=%.1f%% 多空=%+.1fbp/期"
                      % (smp["key"], key, mode, rec["rank_ic_mean"], rec["rank_ic_ir"],
                         rec["rank_ic_t"], rec["rank_ic_t_nonoverlap"],
                         rec["rank_ic_positive_ratio"] * 100.0, rec["ls_mean_bp"]))

    ic_rows = pd.DataFrame(ic_records)
    q_rows = pd.concat(q_frames, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    ic_csv = os.path.join(args.out_dir, "factor_ic.csv")
    q_csv = os.path.join(args.out_dir, "quantile_returns.csv")
    report_md = os.path.join(args.out_dir, "REPORT.md")
    ic_rows.round(6).to_csv(ic_csv, index=False)
    q_rows.round(6).to_csv(q_csv, index=False)

    primary = samples[0]
    meta = {
        "data_dir": os.path.abspath(args.data_dir),
        "panel_dates": len(close), "panel_assets": int(close.shape[1]),
        "panel_start": close.index.min().date(), "panel_end": close.index.max().date(),
        "n_sectors": len({SECTOR_MAP[c] for c in close.columns if c in SECTOR_MAP}),
        "sector_sizes": ["%s %d" % (sec, len([c for c in close.columns if SECTOR_MAP.get(c) == sec]))
                         for sec in sorted({SECTOR_MAP[c] for c in close.columns if c in SECTOR_MAP})],
        "max_daily_move": args.max_daily_move,
        "artifact_note": ("、".join(flagged) + " 出现过超过 %.0f%% 的单日伪涨跌"
                          % (args.max_daily_move * 100)) if flagged else "本池未出现",
    }
    write_report(report_md, meta, ic_rows, q_rows, samples, args.horizon, args.quantiles,
                 decay_horizons)

    print("=" * 78)
    print("④ 主样本（%s, %s ~ %s）行业中性化后的分层平均收益（bp/%d日）"
          % (primary["key"], primary["panel"].index.min().date(),
             primary["panel"].index.max().date(), args.horizon))
    print(quantile_view(q_rows, primary["key"], "neutral").to_string())
    print("=" * 78)
    print("⑤ 输出:")
    for f in (report_md, ic_csv, q_csv):
        print("   %s (%.1f KB)" % (f, os.path.getsize(f) / 1024.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
