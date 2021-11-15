import pytest
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from cartesius.tokenizers import TransformerTokenizer
from cartesius.tokenizers import GraphTokenizer


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
    tokenizer = TransformerTokenizer(max_seq_len=256)
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


@pytest.mark.parametrize("p", [LineString([(0, 0), (1, 1)]), Point((0, 0))])
def test_graph_tokenizer_single_not_polygon(p):
    tokenizer = GraphTokenizer()

    result = tokenizer(p)

    assert result["x"].tolist() == [list(c) for c in p.coords]
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

    assert result["x"].tolist() == [list(c) for c in p[0].boundary.coords[:-1]] + [list(c) for c in p[1].boundary.coords[:-1]]
    assert result["edge_index"].tolist() == [
        [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 6, 5, 3, 6],
        [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 6, 6, 3],
    ]


def test_graph_tokenizer_batched_all_types():
    tokenizer = GraphTokenizer()
    p = [
        Point(0.5, 0.5),
        Polygon([(0, 0), (0, 1), (1, 0), (0, 0)]),
        LineString([(0, 0), (1, 1)]),
    ]

    result = tokenizer(p)

    assert result["x"].tolist() == [list(c) for c in p[0].coords] + [list(c) for c in p[1].boundary.coords[:-1]] + [list(c) for c in p[2].coords]
    assert result["edge_index"].tolist() == [
        [0, 2, 1, 3, 2, 1, 3, 5, 4],
        [0, 1, 2, 2, 3, 3, 1, 4, 5],
    ]
