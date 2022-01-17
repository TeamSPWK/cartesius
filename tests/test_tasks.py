from shapely import wkt
from shapely.geometry import LineString
from shapely.geometry import Point

from cartesius.tasks import GuessAspectRatio
from cartesius.tasks import GuessConvexity
from cartesius.tasks import GuessOmbrRatio
from cartesius.tasks import GuessOpeningRatio


def test_convexity_task_is_close():
    task = GuessConvexity()

    p = wkt.loads(
        "LINESTRING (132.2927559099003 177.2708149869082, 127.0437341469892 171.548151702939, 121.7947123840782 "
        "165.8254884189699, 116.5456906211671 160.1028251350007, 111.2966688582561 154.3801618510315, 106.047647095345 "
        "148.6574985670623, 100.798625332434 142.9348352830931, 79.80253828078979 120.0441821472163)")
    assert p.convex_hull.area != 0

    label = task.get_label(p)
    assert label == 1


def test_guess_ombr_ratio_point():
    task = GuessOmbrRatio()

    p = Point((0, 0))
    label = task.get_label(p)
    assert label == 1


def test_guess_ombr_ratio_linestring_straight():
    task = GuessOmbrRatio()

    p = LineString([(0, 0), (1, 1)])
    label = task.get_label(p)
    assert label == 1


def test_guess_ombr_ratio_linestring_with_angle():
    task = GuessOmbrRatio()

    p = LineString([(0, 0), (1, 0), (1, 1)])
    label = task.get_label(p)
    assert label == 0


def test_guess_aspect_ratio_point():
    task = GuessAspectRatio()

    p = Point((0, 0))
    label = task.get_label(p)
    assert label == 1


def test_guess_aspect_ratio_linestring_straignt():
    task = GuessAspectRatio()

    p = LineString([(0, 0), (1, 1)])
    label = task.get_label(p)
    assert label == 0


def test_guess_aspect_ratio_linestring_with_angle():
    task = GuessAspectRatio()

    p = LineString([(0, 0), (1, 0), (1, 1)])
    label = task.get_label(p)
    assert label > 0


def test_guess_opening_ratio_point():
    task = GuessOpeningRatio()

    p = Point((0, 0))
    label = task.get_label(p)
    assert label == 0


def test_guess_opening_ratio_linestring_straight():
    task = GuessOpeningRatio()

    p = LineString([(0, 0), (1, 1)])
    label = task.get_label(p)
    assert label == 0


def test_guess_opening_ratio_linestring_with_angle():
    task = GuessOpeningRatio()

    p = LineString([(0, 0), (1, 0), (1, 1)])
    label = task.get_label(p)
    assert label == 0
