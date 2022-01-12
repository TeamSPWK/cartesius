import numpy as np
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from cartesius.tokenizers import GraphPolarTokenizer
from cartesius.tokenizers import GraphTokenizer
from cartesius.tokenizers import TransformerAugmentTokenizer
from cartesius.tokenizers import TransformerPolarTokenizer
from cartesius.tokenizers import TransformerTokenizer


def toleq(tgt, ref, tol=1e-4):
    return ref * (1 - tol) < tgt < ref * (1 + tol)


def polar_area(polar_coords):
    polar_arr = np.array(polar_coords)
    theta_arr, r_arr = np.swapaxes(polar_arr, 0, 1)
    theta_diff_arr = theta_arr[1:] - theta_arr[:-1]
    sin_arr = np.sin(theta_diff_arr)
    area = 0
    for i in range(len(sin_arr)):
        area += r_arr[i] * r_arr[i + 1] * sin_arr[i] / 2
    return np.abs(area)


def test_transformer_tokenizer_single_polygon():
    tokenizer = TransformerTokenizer(max_seq_len=256)
    p = Polygon([(0, 0), (0, 1), (1, 0), (0, 0)])

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(4)]
    assert result["polygon"][0].tolist() == [list(c) for c in p.boundary.coords]


@pytest.mark.parametrize("p", [LineString([(0, 0), (1, 1)]), Point((0, 0))])
def test_transformer_tokenizer_single_not_polygon(p):
    tokenizer = TransformerTokenizer(max_seq_len=256)

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(len(p.coords))]
    assert result["polygon"][0].tolist() == [list(c) for c in p.coords]


def test_transformer_tokenizer_batched_polygons():
    tokenizer = TransformerTokenizer(max_seq_len=256)
    p = [
        Polygon([(0, 0), (0, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
    ]

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(4)] + [False]
    assert result["polygon"][0].tolist() == [list(c) for c in p[0].boundary.coords] + [[0, 0]]
    assert result["mask"][1].tolist() == [True for _ in range(5)]
    assert result["polygon"][1].tolist() == [list(c) for c in p[1].boundary.coords]


def test_transformer_tokenizer_too_much_points():
    tokenizer = TransformerTokenizer(max_seq_len=32)
    p = Point((0, 0)).buffer(1, resolution=64)

    with pytest.raises(RuntimeError):
        tokenizer(p)


def test_graph_tokenizer_single_polygon():
    tokenizer = GraphTokenizer()
    p = Polygon([(0, 0), (0, 1), (1, 0), (0, 0)])

    result = tokenizer(p)

    print(result["x"])

    assert result["x"].tolist() == [list(c) for c in p.boundary.coords[:-1]]
    assert result["edge_index"].tolist() == [
        [1, 0, 2, 1, 0, 2],
        [0, 1, 1, 2, 2, 0],
    ]
    assert result["batch_index"].tolist() == [0 for _ in range(len(result["x"]))]


@pytest.mark.parametrize("p", [LineString([(0, 0), (1, 1)]), Point((0, 0))])
def test_graph_tokenizer_single_not_polygon(p):
    tokenizer = GraphTokenizer()

    result = tokenizer(p)

    assert result["x"].tolist() == [list(c) for c in p.coords]
    assert result["batch_index"].tolist() == [0 for _ in range(len(result["x"]))]
    if isinstance(p, LineString):
        assert result["edge_index"].tolist() == [
            [1, 0],
            [0, 1],
        ]
    else:
        assert result["edge_index"].tolist() == [[0], [0]]


def test_graph_tokenizer_batched_polygons():
    tokenizer = GraphTokenizer()
    p = [
        Polygon([(0, 0), (0, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
    ]

    result = tokenizer(p)

    assert result["x"].tolist() == [list(c) for c in p[0].boundary.coords[:-1]
                                   ] + [list(c) for c in p[1].boundary.coords[:-1]]
    assert result["edge_index"].tolist() == [
        [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 6, 5, 3, 6],
        [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 6, 6, 3],
    ]
    assert result["batch_index"].tolist() == [0 for _ in range(3)] + [1 for _ in range(4)]


def test_graph_tokenizer_batched_all_types():
    tokenizer = GraphTokenizer()
    p = [
        Point(0.5, 0.5),
        Polygon([(0, 0), (0, 1), (1, 0), (0, 0)]),
        LineString([(0, 0), (1, 1)]),
    ]

    result = tokenizer(p)

    assert result["x"].tolist() == [list(c) for c in p[0].coords] + [list(c) for c in p[1].boundary.coords[:-1]
                                                                    ] + [list(c) for c in p[2].coords]
    assert result["edge_index"].tolist() == [
        [0, 2, 1, 3, 2, 1, 3, 5, 4],
        [0, 1, 2, 2, 3, 3, 1, 4, 5],
    ]
    assert result["batch_index"].tolist() == [0] + [1 for _ in range(3)] + [2 for _ in range(2)]


def test_transformer_augment_tokenizer_single_polygon():
    tokenizer = TransformerAugmentTokenizer(max_seq_len=256)
    p = Polygon([(0, 0), (0, 1), (1, 0), (0, 0)])

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(256)]
    assert toleq(Polygon(result["polygon"][0].tolist()).area, p.area)


@pytest.mark.parametrize("p", [LineString([(0, 0), (1, 1)]), Point((0, 0))])
def test_transformer_augment_tokenizer_single_not_polygon(p):
    tokenizer = TransformerAugmentTokenizer(max_seq_len=256)

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(256)]
    if isinstance(p, Polygon):
        assert toleq(Polygon(result["polygon"][0].tolist()).area, p.area)
    if isinstance(p, Point):
        assert result["polygon"][0].tolist() == [list(p.coords[0]) for _ in range(256)]


def test_transformer_augment_tokenizer_batched_polygons():
    tokenizer = TransformerAugmentTokenizer(max_seq_len=256)
    p = [
        Polygon([(0, 0), (0, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
    ]

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(256)]
    assert toleq(Polygon(result["polygon"][0].tolist()).area, p[0].area)
    assert result["mask"][1].tolist() == [True for _ in range(256)]
    assert toleq(Polygon(result["polygon"][1].tolist()).area, p[1].area)


def test_transformer_polar_tokenizer_single_polygon():
    tokenizer = TransformerPolarTokenizer(max_seq_len=256, append_original=False)
    p = Polygon([(0, 0), (0, 1), (1, 0), (0, 0)])

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(4)]
    assert toleq(polar_area(result["polygon"][0]), p.area)


@pytest.mark.parametrize("p", [LineString([(0, 0), (1, 1)]), Point((0, 0))])
def test_transformer_polar_tokenizer_single_not_polygon(p):
    tokenizer = TransformerPolarTokenizer(max_seq_len=256, append_original=False)

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(len(p.coords))]


def test_transformer_polar_tokenizer_batched_polygons():
    tokenizer = TransformerPolarTokenizer(max_seq_len=256, append_original=False)
    p = [
        Polygon([(0, 0), (0, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
    ]

    result = tokenizer(p)

    assert result["mask"][0].tolist() == [True for _ in range(4)] + [False]
    assert toleq(polar_area(result["polygon"][0]), p[0].area)
    assert result["mask"][1].tolist() == [True for _ in range(5)]
    assert toleq(polar_area(result["polygon"][1]), p[1].area)


def test_transformer_polar_tokenizer_too_much_points():
    tokenizer = TransformerPolarTokenizer(max_seq_len=32, append_original=False)
    p = Point((0, 0)).buffer(1, resolution=64)

    with pytest.raises(RuntimeError):
        tokenizer(p)


def test_graph_polar_tokenizer_single_polygon():
    tokenizer = GraphPolarTokenizer(append_original=False)
    p = Polygon([(0, 0), (0, 1), (1, 0), (0, 0)])

    result = tokenizer(p)

    assert toleq(polar_area(result["x"]), p.area)
    assert result["edge_index"].tolist() == [
        [1, 0, 2, 1, 0, 2],
        [0, 1, 1, 2, 2, 0],
    ]
    assert result["batch_index"].tolist() == [0 for _ in range(len(result["x"]))]


@pytest.mark.parametrize("p", [LineString([(0, 0), (1, 1)]), Point((0, 0))])
def test_graph_polar_tokenizer_single_not_polygon(p):
    tokenizer = GraphPolarTokenizer(append_original=False)

    result = tokenizer(p)

    assert result["batch_index"].tolist() == [0 for _ in range(len(result["x"]))]
    if isinstance(p, LineString):
        assert result["edge_index"].tolist() == [
            [1, 0],
            [0, 1],
        ]
    else:
        assert result["edge_index"].tolist() == [[0], [0]]


def test_graph_polar_tokenizer_batched_polygons():
    tokenizer = GraphPolarTokenizer(append_original=False)
    p = [
        Polygon([(0, 0), (0, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
    ]

    result = tokenizer(p)

    assert toleq(polar_area(result["x"][:3]), p[0].area)
    assert toleq(polar_area(result["x"][3:]), p[1].area)

    assert result["edge_index"].tolist() == [
        [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 6, 5, 3, 6],
        [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 6, 6, 3],
    ]
    assert result["batch_index"].tolist() == [0 for _ in range(3)] + [1 for _ in range(4)]


def test_graph_polar_tokenizer_batched_all_types():
    tokenizer = GraphPolarTokenizer(append_original=False)
    p = [
        Point(0.5, 0.5),
        Polygon([(0, 0), (0, 1), (1, 0), (0, 0)]),
        LineString([(0, 0), (1, 1)]),
    ]

    result = tokenizer(p)

    assert toleq(polar_area(result["x"][1:4]), p[1].area)
    assert result["edge_index"].tolist() == [
        [0, 2, 1, 3, 2, 1, 3, 5, 4],
        [0, 1, 2, 2, 3, 3, 1, 4, 5],
    ]
    assert result["batch_index"].tolist() == [0] + [1 for _ in range(3)] + [2 for _ in range(2)]
