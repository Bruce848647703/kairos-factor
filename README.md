# Kairos Factor

> Kairos 量化系列的因子研究模块 —— 一个**自研、轻量、零重型依赖**的 Python 因子/Alpha 分析库。

`kairos_factor` 覆盖单因子研究的完整链路：预处理（去极值/标准化/中性化）→ 评价指标
（IC/IR、分层收益、多空、换手）→ 综合报告（IC 衰减、FactorReport），
专为 A 股 / 通用资产的截面因子研究设计。核心代码全部原创，仅依赖 `numpy` 与 `pandas`。

**数据约定**：因子面板与远期收益面板均为 `pandas.DataFrame`，index=日期，columns=资产；
远期收益行 `t` 表示 `(t, t+h]` 区间实现的收益，与因子行 `t` 直接配对，天然防未来函数。

## 特性
- **预处理 `preprocess`**：MAD/分位数去极值 `winsorize`、截面标准化 `zscore`、
  排名归一（均匀或正态分位，自带 numpy 版正态分位函数，不依赖 scipy）`rank_normalize`、
  截面缺失填充 `fill_cross_section`、行业+市值中性化 `neutralize`（`numpy.linalg.lstsq` 取残差）。
- **评价指标 `metrics`**：逐期皮尔逊 IC / 斯皮尔曼秩 IC、IC 汇总（均值/标准差/IR/t 值/胜率）、
  分层（分位）收益 `quantile_returns`、多空收益 `long_short_returns`、覆盖率、因子自相关、换手率。
- **综合评估 `evaluation`**：多期限 IC 衰减 `decay_analysis`；一站式 `FactorReport`
  （dataclass），支持 `to_dict()` / `summary_frame()` 导出。
- **流水线 `pipeline`**：`FactorPipeline` 链式追加截面变换（winsorize → neutralize → zscore），
  保证研究/生产预处理一致。
- **合成数据 `sim`**：`make_factor_and_returns` 生成与远期收益具有**可控真实 IC** 的
  因子面板，示例与测试全部离线、固定 seed 可复现。
- **真实数据 `realdata`**：`load_close_panel` 从本地 CSV 目录读出对齐的 A 股收盘价面板
  （非正价→NaN→历史 ffill→按全体上市日裁剪），配套前复权伪影筛查与行业标签构造，
  完全离线；`examples/real_factor_study.py` 用它在真实行情上出研究报告。
- **逐截面变换**：所有预处理只使用当期截面信息，不引入任何跨期数据。

## 安装
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # 或 pip install numpy pandas
pip install -e ".[dev]"     # 需要跑测试时
```

## 快速开始
### ① 预处理流水线
```python
import kairos_factor as kf
from kairos_factor import FactorPipeline

pipe = FactorPipeline.standard(industry_dummies=labels, log_mktcap=log_mktcap)
clean = pipe.fit_transform(raw_factor)   # winsorize(MAD) -> 行业市值中性化 -> zscore
```

### ② 因子评价报告
```python
from kairos_factor import FactorReport, make_factor_and_returns

factor, fwd = make_factor_and_returns(n_dates=400, n_assets=120, true_ic=0.08, seed=42)
report = FactorReport.build(factor, fwd, n_quantiles=5,
                            prices_or_returns=fwd.shift(1), horizons=(1, 2, 3, 5, 10),
                            kind="return")
print(report.summary_frame())            # IC/秩IC/IR/t 值/多空/覆盖率 标量汇总
print(report.quantile_returns.mean())    # 分层平均收益（应随分位递增）
print(report.decay)                      # IC 随期限衰减
```

完整可运行示例见 [`examples/demo.py`](examples/demo.py)。

## 真实数据因子研究
[`examples/real_factor_study.py`](examples/real_factor_study.py) 在**真实 A 股日线**上跑完整链路：
加载价格 → 构造因子（60日动量 / 20日动量 / 5日反转 / 低波动）→ `FactorPipeline` 预处理
（winsorize + zscore，可选行业中性化）→ `forward_returns` + `rank_ic` / `ic_summary` /
`quantile_returns` / `decay_analysis`（`FactorReport`）→ 落盘中文报告。结论由数据说话，
弱就写弱，不做美化。

```bash
python examples/real_factor_study.py --data-dir /path/to/ashare_csv_dir
# 可选参数：--horizon 5 --quantiles 5 --decay-horizons 1,2,5,10,20
#           --max-daily-move 0.30 --out-dir research/real_factor
```

产物（默认写入 `research/real_factor/`，报告与小 CSV 随仓库入库）：

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 每个因子的 IC 均值 / ICIR / t 值（重叠 + 非重叠两种口径）/ IC>0 占比 / 分层收益 / IC 衰减、数据质量处理说明、诚实结论与局限 |
| `factor_ic.csv` | 因子 × 样本 × 预处理 的全部标量指标（含各期限衰减） |
| `quantile_returns.csv` | 分层组合与多空（LS）的均值、年化、胜率、期数 |

数据加载器 `kairos_factor.realdata`（只读本地 CSV，不联网）：

- `load_close_panel(data_dir, drop_incomplete=True)`：`<symbol>.csv` 目录 → 收盘价面板
  （index=交易日, columns=symbol）。口径与本系列其它加载器一致：非正价（停牌记 0、
  前复权累计调整失真为负）→ NaN → 历史 ffill（无未来函数）；`drop_incomplete=True` 时
  裁掉「任一标的尚未上市」的早期行。
- `artifact_flags` / `clean_start` / `drop_artifact_symbols` / `flagged_symbols`：
  前复权价格趋零会造成数倍乃至数十倍的**伪涨跌**（本池 4 只高分红标的即如此），
  用「|单日涨跌| > 30%（远超 A 股 ±10%/±20% 涨跌停）」筛查，并给出两个互补样本
  （全体标的·共同可信起点 / 剔除失真标的·完整窗口）检验结论是否依赖窗口。
- `sector_labels(index, columns)`：行业分组 → 可直接喂给 `neutralize` 的行业标签面板。

**数据声明**：示例行情来自公开行情接口的前复权日线，仅用于研究与教学演示，
版权归原作者 / 数据源所有，不用于商业用途，不主张对数据本身的所有权；数据不保证准确完整，
报告中的一切数字均为**样本内历史统计特征，不构成投资建议**。

## API 概览
| 模块 | 关键对象 | 说明 |
|---|---|---|
| `preprocess` | `winsorize` `zscore` `rank_normalize` `fill_cross_section` `neutralize` | 逐截面预处理 |
| `metrics` | `ic` `rank_ic` `ic_summary` `quantile_groups` `quantile_returns` `long_short_returns` `factor_coverage` `factor_autocorrelation` `turnover` | 评价指标 |
| `evaluation` | `forward_returns` `decay_analysis` `FactorReport` | 远期收益推算 / IC 衰减 / 报告 |
| `pipeline` | `FactorPipeline` `PipelineStep` | 链式变换流水线 |
| `sim` | `make_factor_and_returns` `make_style_exposures` | 可控 IC 的合成数据 |
| `realdata` | `load_close_panel` `artifact_flags` `clean_start` `drop_artifact_symbols` `flagged_symbols` `sector_labels` | 真实 A 股行情加载 / 前复权伪影筛查 / 行业标签（离线） |

## 设计要点
- **防未来函数**：因子行 `t` 只与「`t` 之后实现」的远期收益配对；`forward_returns`
  由价格或期间收益面板推算 `(t, t+h]` 区间收益，绝不回看。
- **缺失值语义一致**：截面变换保持 NaN 位置不变；相关性计算按「成对有效」剔除；
  样本不足的截面输出 NaN 而非虚假数值。
- **退化情形有约定**：MAD=0 的截面不裁剪；零方差截面 zscore 输出 0；
  中性化时有效样本数不足的截面整体置 NaN，避免欠定方程组的全零假残差。
- **共线性安全**：中性化回归元含截距与全量行业哑变量（秩亏），`lstsq`
  给出最小范数解，残差（正交投影）唯一且正确。
- **无 scipy 硬依赖**：正态分位映射用自研有理逼近 + 牛顿精化实现，精度达机器级。

## 测试
```bash
make test          # 或 python -m pytest -q
```

## 项目结构
```
kairos_factor/      核心包（preprocess / metrics / evaluation / pipeline / sim / realdata）
examples/           可运行示例（demo.py 合成数据 / real_factor_study.py 真实数据）
tests/              pytest 测试（全部离线）
research/           示例产出的研究结果（real_factor/ 报告与 CSV 入库）
```

## 许可
MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 参考与致谢
本项目为**独立原创实现**，未复制任何第三方代码。设计思路受业界通用因子研究范式
（IC/IR 分析、分层（分位）回测与多空组合、去极值/标准化/中性化预处理、IC 衰减分析）
启发，在此向开源量化社区致谢。算法与接口均为本仓库自研。
