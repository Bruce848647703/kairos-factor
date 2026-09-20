"""因子变换流水线。

FactorPipeline 把多个截面变换步骤（如 winsorize → neutralize → zscore）
串成可复用的处理链，保证训练/研究/生产使用完全一致的预处理顺序。

所有内置变换均无状态（逐截面独立计算），因此 fit_transform 与 transform
等价；保留 fit_transform 接口是为了兼容常见流水线使用习惯，并允许未来
挂接需要拟合参数的自定义步骤。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from .preprocess import neutralize, winsorize, zscore

Panel = pd.DataFrame
StepFunc = Callable[..., Panel]


@dataclass
class PipelineStep:
    """流水线中的一步：函数 + 固定参数。"""

    name: str
    func: StepFunc
    params: Dict[str, Any] = field(default_factory=dict)

    def apply(self, panel: Panel) -> Panel:
        out = self.func(panel, **self.params)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"步骤 {self.name!r} 必须返回 DataFrame，实际得到 {type(out)}")
        return out


class FactorPipeline:
    """链式截面变换流水线。

    用法
    ----
    >>> pipe = FactorPipeline()
    >>> pipe.add(winsorize, method="mad").add(zscore)   # 可链式追加
    >>> clean = pipe.fit_transform(raw_factor)

    也支持装饰器注册自定义步骤：

    >>> @pipe.step("demean")
    ... def demean(panel):
    ...     return panel.sub(panel.mean(axis=1), axis=0)
    """

    def __init__(self, steps: Optional[Iterable[PipelineStep]] = None):
        self.steps: List[PipelineStep] = list(steps) if steps else []

    # ---- 构建 ----
    def add(self, func: StepFunc, name: Optional[str] = None, **params) -> "FactorPipeline":
        """追加一个变换步骤（返回自身，支持链式调用）。"""
        if not callable(func):
            raise TypeError("func 必须是可调用对象")
        self.steps.append(PipelineStep(name=name or getattr(func, "__name__", "step"),
                                       func=func, params=params))
        return self

    def step(self, name: Optional[str] = None, **params) -> Callable[[StepFunc], StepFunc]:
        """装饰器形式的 add，被装饰函数原样返回。"""

        def decorator(func: StepFunc) -> StepFunc:
            self.add(func, name=name, **params)
            return func

        return decorator

    @classmethod
    def standard(cls, industry_dummies=None, log_mktcap: Optional[Panel] = None,
                 winsor_method: str = "mad", n_mad: float = 3.0,
                 quantiles=(0.01, 0.99)) -> "FactorPipeline":
        """常用标准流水线：去极值 → （可选）中性化 → 标准化。"""
        pipe = cls()
        pipe.add(winsorize, name="winsorize", method=winsor_method,
                 n_mad=n_mad, quantiles=quantiles)
        if industry_dummies is not None or log_mktcap is not None:
            pipe.add(neutralize, name="neutralize",
                     industry_dummies=industry_dummies, log_mktcap=log_mktcap)
        pipe.add(zscore, name="zscore")
        return pipe

    # ---- 执行 ----
    def transform(self, factor: Panel) -> Panel:
        """按注册顺序对因子面板依次施加全部步骤，返回新面板。"""
        if not isinstance(factor, pd.DataFrame):
            raise TypeError("factor 必须是 pandas.DataFrame（index=日期, columns=资产）")
        out = factor
        for step in self.steps:
            out = step.apply(out)
        return out

    def fit_transform(self, factor: Panel) -> Panel:
        """与 transform 等价（内置步骤无状态、无跨期参数需要拟合）。"""
        return self.transform(factor)

    # ---- 检视 ----
    @property
    def names(self) -> List[str]:
        return [s.name for s in self.steps]

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"FactorPipeline({' -> '.join(self.names) or 'empty'})"
