from math import isclose

import pytest
from shapely.geometry import box
from shapely.geometry import LineString
from shapely.geometry import Point

from cartesius.transforms import NormalizePositionTransform
from cartesius.transforms import NormalizeScaleStaticTransform
from cartesius.transforms import NormalizeScaleTransform


@pytest.mark.parametrize("p", [
    box(1, 1, 10, 10),
    box(1, -10, 10, -1),
    box(-10, 1, -1, 10),
    box(-10, -10, -1, -1),
    box(1, -10, 10, 10),
    box(-10, 1, 10, 10),
    box(-10, -10, 10, 10),
])
def test_norm_pos(p):
    transfo = NormalizePositionTransform()

    result = transfo(p)

    min_x, min_y, *_ = result.bounds
    assert min_x == 0 and min_y == 0
    assert result.area == p.area


@pytest.mark.parametrize("p", [
    box(1, 1, 10, 10),
    box(1, 1, 10, 5),
    box(1, 1, 5, 10),
    Point((1, 1)),
    LineString([(1, 1), (10, 10)]),
    box(1, 1, 1.2, 1.1),
])
def test_norm_scale(p):
    transfo = NormalizeScaleTransform()
    min_x, min_y, max_x, max_y = p.bounds
    scale_size = max([max_x - min_x, max_y - min_y])

    result = transfo(p)

    p_min_x, p_min_y, *_ = p.bounds
    r_min_x, r_min_y, *_ = result.bounds
    assert p_min_x == r_min_x and p_min_y == r_min_y
    if scale_size != 0:
        scale_ratio = 1 / scale_size
        assert isclose(result.area, p.area * scale_ratio**2)


@pytest.mark.parametrize("p", [
    box(1, 1, 10, 10),
    box(1, 1, 10, 5),
    box(1, 1, 5, 10),
    Point((1, 1)),
    LineString([(1, 1), (10, 10)]),
    box(1, 1, 1.2, 1.1),
])
def test_norm_static_scale(p):
    transfo = NormalizeScaleStaticTransform(max_radius_range=5)

    result = transfo(p)

    p_min_x, p_min_y, *_ = p.bounds
    r_min_x, r_min_y, *_ = result.bounds
    assert p_min_x == r_min_x and p_min_y == r_min_y
    scale_ratio = 0.05
    assert isclose(result.area, p.area * scale_ratio**2)
