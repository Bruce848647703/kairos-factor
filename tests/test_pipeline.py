import numpy as np
import pandas as pd
import pytest

import kairos_factor as kf
from kairos_factor import FactorPipeline


def _case(seed=4, n_dates=30, n_assets=40):
    factor, _ = kf.make_factor_and_returns(n_dates=n_dates, n_assets=n_assets,
                                           true_ic=0.1, seed=seed)
    labels, cap = kf.make_style_exposures(n_dates=n_dates, n_assets=n_assets,
                                          n_industries=4, seed=seed + 1)
    contaminated = factor + 0.6 * kf.zscore(cap)
    return contaminated, labels, cap


def test_pipeline_standard_matches_manual_chain():
    f, labels, cap = _case()
    pipe = FactorPipeline.standard(industry_dummies=labels, log_mktcap=cap)
    manual = kf.zscore(kf.neutralize(kf.winsorize(f, method="mad"),
                                     labels, cap))
    pd.testing.assert_frame_equal(pipe.transform(f), manual)
    pd.testing.assert_frame_equal(pipe.fit_transform(f), pipe.transform(f))
    assert pipe.names == ["winsorize", "neutralize", "zscore"]
    assert len(pipe) == 3
    assert "winsorize" in repr(pipe)


def test_pipeline_standard_without_neutralize():
    f, _, _ = _case()
    pipe = FactorPipeline.standard()
    manual = kf.zscore(kf.winsorize(f, method="mad"))
    pd.testing.assert_frame_equal(pipe.transform(f), manual)
    assert pipe.names == ["winsorize", "zscore"]


def test_pipeline_params_passthrough():
    f, _, _ = _case()
    pipe = FactorPipeline().add(kf.winsorize, method="quantile",
                                quantiles=(0.2, 0.8))
    manual = kf.winsorize(f, method="quantile", quantiles=(0.2, 0.8))
    pd.testing.assert_frame_equal(pipe.transform(f), manual)


def test_pipeline_chaining_and_decorator():
    f, _, _ = _case()

    @kf.FactorPipeline().step("demean")
    def demean(panel):
        return panel.sub(panel.mean(axis=1), axis=0)

    pipe = FactorPipeline().add(kf.winsorize).add(demean, name="demean")
    out = pipe.transform(f)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-12)
    assert pipe.names == ["winsorize", "demean"]

    # 装饰器注册后原函数仍可独立调用，且与流水线结果一致
    assert demean(kf.winsorize(f)).equals(pipe.transform(f))


def test_pipeline_empty_returns_input_values():
    f, _, _ = _case()
    pipe = FactorPipeline()
    pd.testing.assert_frame_equal(pipe.transform(f), f.astype("float64"))


def test_pipeline_validates_step_output_and_input():
    f, _, _ = _case()
    pipe = FactorPipeline().add(lambda panel: panel.iloc[0], name="bad")
    with pytest.raises(TypeError):
        pipe.transform(f)
    with pytest.raises(TypeError):
        FactorPipeline().transform([1, 2, 3])
    with pytest.raises(TypeError):
        FactorPipeline().add("not_callable")
