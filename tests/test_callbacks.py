import numpy as np
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from cartesius.callbacks import TransformerHardSamplePlotCallback


@pytest.mark.parametrize("p", [Polygon([(0, 0), (0, 1), (1, 0)]), LineString([(0, 0), (1, 1)]), Point((0, 0))])
def test_poly_coords_from_feature(p):
    p_coords = list(p.boundary.coords) if isinstance(p, Polygon) else list(p.coords)
    pad_size = 10
    m = [1 if i < len(p_coords) else 0 for i in range(pad_size)]
    p = p_coords + [(0, 0) for _ in range(pad_size - len(p_coords))]
    x = {"polygon": np.array(p), "mask": np.array(m)}
    poly_coords_from_feature = TransformerHardSamplePlotCallback.poly_coords_from_feature
    coords = poly_coords_from_feature(x)
    assert np.equal(p_coords, coords).all()
