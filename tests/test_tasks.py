import pytest
from shapely import wkt

from cartesius.tasks import GuessConvexity, GuessOmbrRatio, GuessOpeningRatio, GuessAspectRatio


def test_convexity_task_is_close():
    task = GuessConvexity()

    p = wkt.loads(
        "LINESTRING (132.2927559099003 177.2708149869082, 127.0437341469892 171.548151702939, 121.7947123840782 "
        "165.8254884189699, 116.5456906211671 160.1028251350007, 111.2966688582561 154.3801618510315, 106.047647095345 "
        "148.6574985670623, 100.798625332434 142.9348352830931, 79.80253828078979 120.0441821472163)")
    assert p.convex_hull.area != 0

    label = task.get_label(p)
    assert label == 1


def test_ombr_task_div_by_zero():
    task = GuessOmbrRatio()

    p = wkt.loads("LINESTRING (0 0, 1 1)")
    assert p.minimum_rotated_rectangle.area == 0

    label = task.get_label(p)
    assert label == 1


def test_opening_task_div_by_zero():
    task = GuessOpeningRatio()

    p = wkt.loads("LINESTRING (0 0, 1 1)")
    assert p.area == 0

    label = task.get_label(p)
    assert label == 0


def test_aspect_task_div_by_zero():
    task = GuessAspectRatio()

    p = wkt.loads("LINESTRING (0 0, 1 1)")
    assert p.area == 0

    label = task.get_label(p)
    assert label == 1
