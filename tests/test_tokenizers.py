import pytest
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from cartesius.tokenizers import Tokenizer


def test_tokenizer_single_polygon():
    tokenizer = Tokenizer(max_seq_len=256)
    p = Polygon([(0, 0), (0, 1), (1, 0), (0, 0)])

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(4)]
    assert result["polygon"][0].tolist() == [list(c) for c in p.boundary.coords]


@pytest.mark.parametrize("p", [LineString([(0, 0), (1, 1)]), Point((0, 0))])
def test_tokenizer_single_not_polygon(p):
    tokenizer = Tokenizer(max_seq_len=256)

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(len(p.coords))]
    assert result["polygon"][0].tolist() == [list(c) for c in p.coords]


def test_tokenizer_batched_polygons():
    tokenizer = Tokenizer(max_seq_len=256)
    p = [
        Polygon([(0, 0), (0, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
    ]

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(4)] + [False]
    assert result["polygon"][0].tolist() == [list(c) for c in p[0].boundary.coords] + [[0, 0]]
    assert result["mask"][1].tolist() == [True for _ in range(5)]
    assert result["polygon"][1].tolist() == [list(c) for c in p[1].boundary.coords]


def test_tokenizer_too_much_points():
    tokenizer = Tokenizer(max_seq_len=256)
    p = Point((0, 0)).buffer(1, resolution=64)

    with pytest.raises(RuntimeError):
        tokenizer(p)
