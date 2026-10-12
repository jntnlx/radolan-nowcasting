"""Units and transforms: the conversions every result depends on."""

import numpy as np
import torch

from nowcast.config import index_time, season_length, time_index
from nowcast.radklim import NODATA, depth_to_rate
from nowcast.transforms import pool_rate, rate_to_y, y_to_rate


def test_depth_integers_to_rate():
    depth = np.array([0, 1, 100, NODATA], dtype=np.uint16)  # 0, 0.01, 1.0 mm per 5 min
    rate = depth_to_rate(depth)
    assert np.allclose(rate[:3], [0.0, 0.12, 12.0]) and np.isnan(rate[3])


def test_transform_round_trip_and_dry_is_zero():
    rate = torch.tensor([0.0, 0.12, 1.0, 10.0, 300.0])
    assert torch.allclose(y_to_rate(rate_to_y(rate)), rate, rtol=1e-5)
    assert rate_to_y(torch.tensor(0.0)) == 0


def test_pooling_averages_valid_rates_only():
    rate = torch.tensor([[4.0, 0.0], [2.0, 99.0]])
    valid = torch.tensor([[True, True], [True, False]])
    pooled, ok = pool_rate(rate, valid, 2)
    assert torch.allclose(pooled, torch.tensor([[2.0]])) and bool(ok)


def test_time_axis():
    from datetime import datetime
    assert season_length(2021) == 214 * 288
    assert time_index(datetime(2021, 4, 1, 0, 5)) == (2021, 1)
    assert index_time(*time_index(datetime(2021, 7, 14, 18, 30))) == datetime(2021, 7, 14, 18, 30)
