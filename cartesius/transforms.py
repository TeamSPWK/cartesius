from shapely.affinity import scale
from shapely.affinity import translate
from shapely.geometry import Point


class NormalizePositionTransform:
    """Transform that normalize the position of the given polygon, so that the
    polygon is always is located in (0, 0), with positive coordinates.
    """

    def __call__(self, polygon):
        min_x, min_y, *_ = polygon.bounds
        return translate(polygon, -min_x, -min_y)


class NormalizeScaleTransform:
    """Transform that normalize the scale of the given polygon, so that the
    polygon's size is always between 0 and 1.
    """

    def __call__(self, polygon):
        min_x, min_y, max_x, max_y = polygon.bounds
        x_size = max_x - min_x
        y_size = max_y - min_y

        scale_size = max([x_size, y_size])

        if scale_size == 0:
            # Nothing to scale
            return polygon

        ref = Point(min_x, min_y)
        scale_ratio = 1 / scale_size

        return scale(polygon, xfact=scale_ratio, yfact=scale_ratio, origin=ref)


TRANSFORMS = {"norm_pos": NormalizePositionTransform(), "norm_scale": NormalizeScaleTransform()}
