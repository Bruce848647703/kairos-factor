"""真实数据加载器 kairos_factor.realdata 的离线测试（用 tmp_path 造小 CSV，不联网）。"""
import numpy as np
import pandas as pd
import pytest

from kairos_factor import realdata
from kairos_factor.realdata import (
    SECTOR_MAP,
    artifact_flags,
    clean_start,
    drop_artifact_symbols,
    flagged_symbols,
    load_close_panel,
    sector_labels,
)

OHLCV = ["date", "open", "high", "low", "close", "volume"]


def _write_csv(directory, symbol, closes, start="2021-01-01", freq="B"):
    """按本系列行情 CSV 口径写一个小文件：date,open,high,low,close,volume。"""
    dates = pd.bdate_range(start, periods=len(closes), freq=freq)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1000.0] * len(closes),
    })[OHLCV]
    path = directory / ("%s.csv" % symbol)
    df.to_csv(str(path), index=False)
    return dates


def _toy_dir(tmp_path):
    """3 个标的小样本：起点不同、含停牌(0)与前复权失真(负价)。"""
    _write_csv(tmp_path, "aaa", [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    _write_csv(tmp_path, "bbb", [20.0, 0.0, -3.0, 22.0, 23.0], start="2021-01-04")
    _write_csv(tmp_path, "ccc", [30.0, 31.0, 32.0, 33.0, 34.0, 35.0])
    return tmp_path


# ---------------------------------------------------------------------------
# load_close_panel
# ---------------------------------------------------------------------------

def test_load_close_panel_shape_and_column_alignment(tmp_path):
    dates = _write_csv(tmp_path, "aaa", [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    _write_csv(tmp_path, "bbb", [20.0, 21.0, 22.0], start="2021-01-04")
    panel = load_close_panel(str(tmp_path), drop_incomplete=False)

    assert list(panel.columns) == ["aaa", "bbb"]          # 按文件名排序，确定性
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert panel.index.is_monotonic_increasing
    assert panel.index.min() == dates[0] and panel.index.max() == dates[-1]
    assert panel.shape == (6, 2)
    assert panel.loc[dates[0], "aaa"] == pytest.approx(10.0)
    assert panel.loc[dates[2], "bbb"] == pytest.approx(21.0)
    # bbb 上市前 -> NaN；上市后无数据的日子按本系列口径用历史价 ffill（不用未来数据）
    assert panel["bbb"].isna().tolist() == [True, False, False, False, False, False]
    assert panel.loc[dates[4], "bbb"] == pytest.approx(22.0)


def test_load_close_panel_nonpositive_and_suspension_handling(tmp_path):
    dates = _write_csv(tmp_path, "aaa", [10.0, 0.0, -3.0, 13.0, np.nan, 15.0])
    panel = load_close_panel(str(tmp_path), drop_incomplete=False)
    got = panel["aaa"].tolist()
    # 非正价（停牌记 0 / 前复权为负）与空值一律视作缺失，用历史值 ffill
    assert got[0] == pytest.approx(10.0)
    assert got[1] == pytest.approx(10.0)
    assert got[2] == pytest.approx(10.0)
    assert got[3] == pytest.approx(13.0)
    assert got[4] == pytest.approx(13.0)
    assert got[5] == pytest.approx(15.0)
    assert not panel.isna().any().any()
    assert dates[1] in panel.index


def test_load_close_panel_leading_invalid_stays_nan(tmp_path):
    # 首部无历史可填 -> 保持 NaN，交由 drop_incomplete 裁剪
    _write_csv(tmp_path, "aaa", [-1.0, 0.0, 12.0, 13.0])
    panel = load_close_panel(str(tmp_path), drop_incomplete=False)
    assert panel["aaa"].isna().tolist() == [True, True, False, False]


def test_load_close_panel_drop_incomplete_crops_to_common_start(tmp_path):
    _toy_dir(tmp_path)
    cropped = load_close_panel(str(tmp_path), drop_incomplete=True)
    full = load_close_panel(str(tmp_path), drop_incomplete=False)

    # bbb 自 2021-01-04 起有价（其停牌/负价日由 ffill 补齐）-> 共同起点为 2021-01-04
    assert cropped.index[0] == pd.Timestamp("2021-01-04")
    assert full.index[0] == pd.Timestamp("2021-01-01")
    assert len(cropped) < len(full)
    assert not cropped.isna().any().any()          # 裁剪后内部无缺失
    assert full.isna().any().any()
    assert list(cropped.columns) == list(full.columns)
    # 裁剪只删行，不改值
    pd.testing.assert_frame_equal(cropped, full.loc[cropped.index[0]:])


def test_load_close_panel_is_deterministic(tmp_path):
    _toy_dir(tmp_path)
    a = load_close_panel(str(tmp_path))
    b = load_close_panel(str(tmp_path))
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["aaa", "bbb", "ccc"]

    # 文件写入顺序不影响列顺序（按文件名排序）
    other = tmp_path / "other"
    other.mkdir()
    for sym in ("ccc", "aaa", "bbb"):
        (tmp_path / ("%s.csv" % sym)).rename(other / ("%s.csv" % sym))
    c = load_close_panel(str(other))
    assert list(c.columns) == list(a.columns)
    pd.testing.assert_frame_equal(c, a)


def test_load_close_panel_duplicate_dates_keep_last(tmp_path):
    path = tmp_path / "aaa.csv"
    pd.DataFrame({
        "date": ["2021-01-01", "2021-01-04", "2021-01-04"],
        "open": [1.0, 2.0, 2.5], "high": [1.0, 2.0, 2.5], "low": [1.0, 2.0, 2.5],
        "close": [10.0, 11.0, 12.0], "volume": [1.0, 1.0, 1.0],
    })[OHLCV].to_csv(str(path), index=False)
    panel = load_close_panel(str(tmp_path))
    assert len(panel) == 2
    assert panel.loc[pd.Timestamp("2021-01-04"), "aaa"] == pytest.approx(12.0)


def test_load_close_panel_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_close_panel(str(tmp_path / "missing"))          # 目录不存在

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "note.txt").write_text("no csv here")
    with pytest.raises(FileNotFoundError):
        load_close_panel(str(empty))                         # 没有 CSV

    bad = tmp_path / "bad"
    bad.mkdir()
    pd.DataFrame({"date": ["2021-01-01"], "price": [1.0]}).to_csv(str(bad / "aaa.csv"),
                                                                  index=False)
    with pytest.raises(ValueError):
        load_close_panel(str(bad))                           # 缺 close 列

    dead = tmp_path / "dead"
    dead.mkdir()
    _write_csv(dead, "aaa", [10.0, 11.0])
    _write_csv(dead, "bbb", [0.0, -1.0])                     # 全列无效
    with pytest.raises(ValueError):
        load_close_panel(str(dead))


def test_load_close_panel_feeds_library(tmp_path):
    """加载结果可直接喂给本库的 forward_returns / rank_ic（口径一致，无未来函数）。"""
    import kairos_factor as kf

    _toy_dir(tmp_path)
    panel = load_close_panel(str(tmp_path))
    fwd = kf.forward_returns(panel, horizon=1, kind="price")
    assert fwd.shape == panel.shape
    assert fwd.iloc[-1].isna().all()                         # 末日无未来数据
    factor = -panel.pct_change(1)
    ic_series = kf.rank_ic(kf.zscore(factor), fwd, min_obs=2)
    assert isinstance(ic_series, pd.Series)
    assert np.isfinite(kf.ic_summary(ic_series)["mean"])


# ---------------------------------------------------------------------------
# 前复权伪影筛查
# ---------------------------------------------------------------------------

def _artifact_panel():
    """构造一个含伪影的面板：bbb 在第 3 日出现 +890% 的伪涨跌，aaa/ccc 正常。"""
    idx = pd.bdate_range("2021-01-01", periods=6)
    aaa = pd.Series([10.0, 10.2, 10.1, 10.3, 10.4, 10.5], index=idx)
    bbb = pd.Series([1.0, 1.01, 10.0, 1.05, 1.06, 1.07], index=idx)
    ccc = pd.Series([50.0, 50.5, 50.2, 50.8, 51.0, 51.2], index=idx)
    return pd.DataFrame({"aaa": aaa, "bbb": bbb, "ccc": ccc})


def test_artifact_flags_marks_implausible_moves():
    panel = _artifact_panel()
    flags = artifact_flags(panel, max_daily_move=0.30)
    assert flags.dtypes.map(lambda d: d == bool).all()
    assert flags["bbb"].tolist() == [False, False, True, True, False, False]
    assert not flags["aaa"].any()
    assert not artifact_flags(panel.iloc[:1], 0.30).any().any()   # 首行无收益


def test_artifact_flags_validates_threshold():
    with pytest.raises(ValueError):
        artifact_flags(_artifact_panel(), max_daily_move=0.0)


def test_clean_start_and_drop_symbols():
    panel = _artifact_panel()
    assert flagged_symbols(panel) == ["bbb"]
    assert clean_start(panel) == panel.index[4]              # 最后一次伪影之后的首个交易日
    kept = drop_artifact_symbols(panel)
    assert list(kept.columns) == ["aaa", "ccc"]
    assert len(kept) == len(panel)

    tidy = panel.drop(columns=["bbb"])
    assert flagged_symbols(tidy) == []
    assert clean_start(tidy) == tidy.index[0]
    pd.testing.assert_frame_equal(drop_artifact_symbols(tidy), tidy)


def test_clean_start_raises_when_artifact_at_end():
    panel = _artifact_panel()
    with pytest.raises(ValueError):
        clean_start(panel.iloc[:3])                          # 伪影延续到最后一日
    with pytest.raises(ValueError):
        drop_artifact_symbols(panel.drop(columns=["aaa"]))   # 剔完只剩 1 个标的


# ---------------------------------------------------------------------------
# 行业标签
# ---------------------------------------------------------------------------

def test_sector_labels_panel():
    idx = pd.bdate_range("2021-01-01", periods=3)
    cols = ["sh600519", "sh600036", "unknown01"]
    labels = sector_labels(idx, cols)
    assert labels.shape == (3, 3)
    assert list(labels.columns) == cols
    assert labels.index.equals(idx)
    assert labels["sh600519"].tolist() == ["consumer"] * 3
    assert labels["sh600036"].tolist() == ["finance"] * 3
    assert labels["unknown01"].isna().all()                  # 未收录标的 -> 缺失


def test_sector_labels_custom_groups_and_map_coverage():
    idx = pd.bdate_range("2021-01-01", periods=2)
    labels = sector_labels(idx, ["x1", "x2"], groups={"g1": ["x1"], "g2": ["x2"]})
    assert labels.iloc[0].tolist() == ["g1", "g2"]
    assert set(SECTOR_MAP) == {s for syms in realdata.SECTOR_GROUPS.values() for s in syms}
    assert len(realdata.SECTOR_GROUPS) == 8


def test_sector_labels_feeds_neutralize():
    """行业标签面板可直接用于 neutralize（有效样本需多于回归元个数）。"""
    import kairos_factor as kf

    idx = pd.bdate_range("2021-01-01", periods=4)
    cols = ["a1", "a2", "a3", "b1", "b2"]
    rng = np.random.default_rng(7)
    panel = pd.DataFrame(100.0 * (1.0 + rng.normal(0.0, 0.01, (4, 5))).cumprod(axis=0),
                         index=idx, columns=cols)
    labels = sector_labels(idx, cols, groups={"ga": ["a1", "a2", "a3"], "gb": ["b1", "b2"]})
    resid = kf.neutralize(panel.pct_change(1), industry_dummies=labels)
    assert resid.shape == panel.shape
    assert resid.iloc[0].isna().all()                        # 首行因子全 NaN
    assert np.isfinite(resid.to_numpy()[1:]).all()
    # 残差在每个截面上与行业哑变量正交：组内均值约等于 0
    tail = resid.iloc[1:]
    assert abs(float(tail[["a1", "a2", "a3"]].mean(axis=1).abs().max())) < 1e-12
    assert abs(float(tail[["b1", "b2"]].mean(axis=1).abs().max())) < 1e-12
