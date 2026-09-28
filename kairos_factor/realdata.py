"""真实 A 股行情加载器（读本地 CSV，完全离线）。

本模块只做「把磁盘上的日线 CSV 变成因子研究可用的面板」这一件事：

- load_close_panel:        CSV 目录 -> 收盘价面板（index=交易日, columns=symbol）；
- artifact_flags:          标记前复权失真导致的「不可能单日涨跌」；
- clean_start:             求全体标的共同的可信起点（此后不再有伪影）；
- drop_artifact_symbols:   剔除历史上出现过伪影的标的，用于保留更长窗口的稳健性检验；
- sector_labels:           行业分组 -> 可直接喂给 neutralize 的行业标签面板。

口径与本系列其它加载器（kairos-data / kairos-strategies）保持一致：
非正价格（停牌记 0、前复权累计调整失真为负）视作缺失 -> NaN，再用历史值 ffill
（只用过去，无未来函数）；drop_incomplete=True 时裁掉「任一标的尚未上市」的早期行，
使面板在每个截面上都有完整样本。

仅依赖 numpy/pandas + 标准库；解析逻辑为本项目原创，测试全部离线。

数据来自公开行情接口，仅用于研究与演示，版权归原作者所有，不构成投资建议。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

Panel = pd.DataFrame

# CSV 约定列（与 kairos-data 抓取产物一致）：date,open,high,low,close,volume；
# 本加载器只使用 date 与 close 两列。
DATE_COLUMN = "date"
PRICE_COLUMN = "close"

# 跨行业流动 A 股池的行业分组（与本系列 universe 一致），用于行业中性化
SECTOR_GROUPS: Dict[str, List[str]] = {
    "consumer": ["sh600519", "sz000858", "sh600887", "sh600809", "sz002304", "sh601888"],
    "finance": ["sh600036", "sh601318", "sh601166", "sh601398", "sh600030", "sz300059"],
    "appliance": ["sz000333", "sz000651", "sh600690"],
    "pharma": ["sh600276", "sz300760", "sh600196"],
    "tech": ["sz002415", "sz002475", "sh603501", "sz300750"],
    "auto_newenergy": ["sz002594", "sh601012", "sh600438", "sh601633"],
    "energy_material": ["sh601088", "sh600028", "sh601899", "sh600585"],
    "industrial": ["sh600031", "sh601766", "sh600900", "sh600009", "sh601668",
                   "sh600048", "sz002714", "sz002352"],
}
SECTOR_MAP: Dict[str, str] = {sym: sec for sec, syms in SECTOR_GROUPS.items() for sym in syms}

# A 股单日涨跌停上界：主板 ±10%、创业板/科创板 ±20%。
# 超过该幅度（含缓冲）的「日收益」只可能来自数据伪影，不可能来自真实成交。
DEFAULT_MAX_DAILY_MOVE = 0.30


def _list_symbol_files(data_dir: str) -> List[str]:
    """列出目录下的 CSV 文件名（排序，保证列顺序确定）。"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 CSV 行情文件")
    return files


def _read_symbol_close(path: str, field: str = PRICE_COLUMN) -> pd.Series:
    """读单个标的 CSV -> 以日期为 index 的收盘价序列（按日期升序、去重）。"""
    df = pd.read_csv(path)
    missing = [c for c in (DATE_COLUMN, field) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必需列: {missing}")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    s = df.set_index(DATE_COLUMN).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return pd.to_numeric(s[field], errors="coerce")


def load_close_panel(data_dir: str, drop_incomplete: bool = True) -> Panel:
    """从本地 CSV 目录加载对齐的收盘价面板。

    参数
    ----
    data_dir:        行情目录，内含若干 ``<symbol>.csv``（列 date,open,high,low,close,volume）；
    drop_incomplete: True 时把窗口起点裁到「全体标的都已上市且价格可信」的第一日，
                     使面板内部无 NaN（便于回测/因子研究）；False 时保留各标的原始起点，
                     未上市区间为 NaN。

    返回 DataFrame：index=交易日(DatetimeIndex, 升序)，columns=symbol（按文件名排序）。

    清洗规则（与本系列其它加载器一致）：
    1. 非正价格（停牌记 0、前复权累计调整导致为负）视作无效 -> NaN；
    2. 用历史值 ffill 填补停牌缺口（只用过去数据，无未来函数）；
    3. drop_incomplete=True 时裁掉任一标的仍无效的早期行。
    """
    series: Dict[str, pd.Series] = {}
    for fname in _list_symbol_files(data_dir):
        symbol = fname[:-len(".csv")]
        series[symbol] = _read_symbol_close(os.path.join(data_dir, fname))

    panel = pd.DataFrame(series).sort_index()
    panel = panel.astype("float64")

    # ① 非正价（停牌记 0 / 前复权失真为负）-> NaN；② 历史 ffill 填补停牌缺口
    panel = panel.where(panel > 0)
    invalid = [c for c in panel.columns if not panel[c].notna().any()]
    if invalid:
        raise ValueError(f"以下标的没有任何有效（正）价格: {invalid}")
    panel = panel.ffill()

    if drop_incomplete:
        first_valid = panel.apply(lambda s: s.first_valid_index())
        start = first_valid.max()
        if pd.isna(start):
            raise ValueError("所有标的均无有效价格，无法确定面板起点")
        panel = panel.loc[start:]
    return panel


# ---------------------------------------------------------------------------
# 前复权伪影筛查
# ---------------------------------------------------------------------------

def artifact_flags(panel: Panel, max_daily_move: float = DEFAULT_MAX_DAILY_MOVE) -> Panel:
    """标记「不可能来自真实成交」的单日涨跌：|日收益| > max_daily_move。

    前复权价 = 原始价 - 累计分红调整，高分红标的的早年价格会被压到接近 0 甚至为负；
    价格趋零时同样的真实涨跌会被放大成数倍乃至数十倍的伪收益。
    返回同形状的 bool 面板（首行无收益、缺失位置均为 False）。
    """
    if max_daily_move <= 0:
        raise ValueError("max_daily_move 必须为正数")
    r = panel.pct_change().abs()
    return (r > float(max_daily_move)).fillna(False)


def clean_start(panel: Panel, max_daily_move: float = DEFAULT_MAX_DAILY_MOVE) -> pd.Timestamp:
    """求「全体标的此后都不再有伪影」的最早交易日。

    做法：对每个标的取其最后一次伪影日期，全局起点 = 该最大值之后的第一个交易日。
    若伪影一直延续到面板最后一日，则不存在可信起点，抛出 ValueError。
    """
    flags = artifact_flags(panel, max_daily_move)
    last = flags.apply(lambda c: c[c].index.max() if bool(c.any()) else pd.NaT).dropna()
    if last.empty:
        return panel.index[0]
    after = panel.index[panel.index > last.max()]
    if len(after) == 0:
        raise ValueError("伪影延续到面板末尾，无法确定可信起点")
    return after[0]


def drop_artifact_symbols(panel: Panel,
                          max_daily_move: float = DEFAULT_MAX_DAILY_MOVE) -> Panel:
    """剔除历史中出现过伪影的标的（整列丢弃），保留尽可能长的样本窗口。

    与 clean_start 互补：clean_start 牺牲窗口长度换取标的完整，
    本函数牺牲少数失真标的换取窗口长度，两者结论一致才说明结果稳健。
    """
    flags = artifact_flags(panel, max_daily_move)
    keep = ~flags.any(axis=0)
    out = panel.loc[:, keep]
    if out.shape[1] < 2:
        raise ValueError(f"剔除伪影标的后仅剩 {out.shape[1]} 个标的，无法做截面研究")
    return out


def flagged_symbols(panel: Panel,
                    max_daily_move: float = DEFAULT_MAX_DAILY_MOVE) -> List[str]:
    """返回被 artifact_flags 判定为失真的标的列表（按列顺序）。"""
    flags = artifact_flags(panel, max_daily_move)
    return [str(c) for c in flags.columns[flags.any(axis=0)]]


# ---------------------------------------------------------------------------
# 行业标签
# ---------------------------------------------------------------------------

def sector_labels(index: Sequence, columns: Sequence[str],
                  groups: Optional[Dict[str, List[str]]] = None) -> Panel:
    """构造 neutralize / FactorPipeline.standard 可直接使用的行业标签面板。

    index=日期，columns=资产，值为行业名（如 "finance"）；未收录的标的为 None。
    行业归属假定在样本期内不变（本池为大盘蓝筹，行业口径稳定）。
    """
    mapping = SECTOR_MAP if groups is None else {
        sym: sec for sec, syms in groups.items() for sym in syms
    }
    cols = list(columns)
    row = np.array([mapping.get(c) for c in cols], dtype=object)
    return pd.DataFrame(np.tile(row, (len(index), 1)), index=index, columns=cols)
