# Generating handcrafted polygons for test/validation

To generate the data used for the test set, run the following script :

```python
import json
import os
import random

from shapely.affinity import rotate
from shapely.affinity import scale
from shapely.affinity import translate
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

MAX_BOUND = 10000
MIN_BOUND = -MAX_BOUND
ANGLE = 360
FILE = "cartesius/data/testset.json"


def move_top_right(p):
    return translate(p, random.random() * MAX_BOUND, random.random() * MAX_BOUND)


def move_top_left(p):
    return translate(p, random.random() * MIN_BOUND, random.random() * MAX_BOUND)


def move_bot_right(p):
    return translate(p, random.random() * MAX_BOUND, random.random() * MIN_BOUND)


def move_bot_left(p):
    return translate(p, random.random() * MIN_BOUND, random.random() * MIN_BOUND)


def scale_big(p):
    s = random.random() * MAX_BOUND
    return scale(p, s, s)


def scale_small(p):
    s = random.random()
    return scale(p, s, s)


def turn(p):
    return rotate(p, random.random() * ANGLE)


def reverse(p):
    prob_x = random.random()
    prob_y = random.random()
    x = p

    if prob_x > 0.5:
        x = scale(x, xfact=-1)
    if prob_y > 0.5:
        x = scale(x, yfact=-1)

    return x


def gen_polygons():
    polygons = []

    # Generate simple Point
    p = Point((0, 0))
    polygons.append(move_top_right(p))
    polygons.append(move_top_left(p))
    polygons.append(move_bot_right(p))
    polygons.append(move_bot_left(p))

    # Generate simple Line
    p = LineString([(0, 0), (1, 1)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate simple Line with multiple points
    p = LineString([(0, 0), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (1, 1)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate half-convex Line
    p = LineString([(0, 0), (0.5, 0.1), (0.6, 0.5), (0.6, 0.9), (0.4, 1), (0, 1)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))
    p = LineString([(0, 0), (0.2, 0.1), (0.6, 1), (0.1, 1)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate almost-polygon Line
    p = LineString([(0.5, 0), (0.75, 0.25), (0.75, 0.5), (0.5, 0.75), (0.25, 0.5), (0.25, 0.25)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))
    p = LineString([(0.5, 0), (0.75, 0.25), (0.75, 0.5), (0.5, 0.75), (0.25, 0.5), (0.25, 0.25), (0, 0.25), (0.1, 0.3)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))
    p = LineString([(0.5, 0), (0.75, 0.5), (0.5, 0.75), (0.25, 0.5), (0.4, 0.4)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate simple triangle
    p = Polygon([(0, 0), (1, 0), (0.2, 0.7), (0, 0)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))
    p = Polygon([(0, 0), (1, 0), (0.5, 1), (0, 0)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate triangle with additional points
    p = Polygon([(0, 0), (0.4, 0), (0.6, 0), (1, 0), (0.5, 1), (0, 0)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate square
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate square with additional points
    p = Polygon([(0, 0), (0.4, 0), (0.6, 0), (1, 0), (1, 0.1), (1, 1), (0, 1), (0, 0.5), (0, 0)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate rectangle
    p = Polygon([(0, 0), (1, 0), (1, 0.6), (0, 0.6), (0, 0)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate rectangle with additional points
    p = Polygon([(0, 0), (0.4, 0), (0.6, 0), (1, 0), (1, 0.1), (1, 0.6), (0, 0.6), (0, 0.3), (0, 0)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate octogon
    p = Point((0.5, 0.5)).buffer(0.5, resolution=2)
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate circle
    p = Point((0.5, 0.5)).buffer(0.5)
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate circle with more points
    p = Point((0.5, 0.5)).buffer(0.5, resolution=32)
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate concave polygon (half circle)
    p = Point((0.5, 0.5)).buffer(0.5,
                                 resolution=8).difference(Polygon([
                                     (-1, -1), (0.5, -1), (0.5, 2), (-1, 2), (-1, -1)
                                 ])).difference(Point((0.5, 0.5)).buffer(0.3, resolution=8))
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate concave polygon (almost full circle)
    p = Point(
        (0.5, 0.5)).buffer(0.5,
                           resolution=10).difference(Polygon([
                               (-1, 0.4), (0.5, 0.4), (0.5, 0.6), (-1, 0.6), (-1, 0.4)
                           ])).difference(Point((0.5, 0.5)).buffer(0.35, resolution=10))
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate long but thin rectangle
    p = Polygon([(0, 0), (1, 0), (1, 0.01), (0, 0.01), (0, 0)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate long but thin rectangle with more points
    p = Polygon([(0, 0), (0.1, 0), (0.2, 0), (0.7, 0), (1, 0), (1, 0.01), (0.55, 0.01), (0, 0.01), (0, 0)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate long but thin rectangle, concave
    p = Polygon([(0, 0), (1, 0), (1, 0.01), (0, 0.01),
                 (0, 0)]).union(Polygon([(0, 0), (0.01, 0), (0.01, 1), (0, 1), (0, 0)]))
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    # Generate something that look like a parcel
    p = Polygon([(0.6, 0), (0.8, 0), (0.8, 0.4), (0.9, 0.4), (0.8, 0.95), (0.1, 0.95), (0.1, 0.4), (0.6, 0.4),
                 (0.6, 0)])
    polygons.append(move_top_right(turn(scale_big(reverse(p)))))
    polygons.append(move_top_right(turn(scale_small(reverse(p)))))
    polygons.append(move_top_left(turn(scale_big(reverse(p)))))
    polygons.append(move_top_left(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_right(turn(scale_small(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_big(reverse(p)))))
    polygons.append(move_bot_left(turn(scale_small(reverse(p)))))

    return polygons


if __name__ == "__main__":
    polygons = gen_polygons()

    data = [p.wkt for p in polygons]
    with open(FILE, "w") as f:
        json.dump(data, f, indent=4)

```

---

The **validation data** is generated with the same script, because this script is **randomized** and produce **different data** every time it's run.

Don't forget to modify the file destination (`FILE` constant).
