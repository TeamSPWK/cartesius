from shapely.affinity import scale
from shapely.affinity import translate
from shapely.geometry import Point


class Transform:
    """Base class for transforms. A Transform is a callable that take a polygon
    as input and transform it appropriately.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, polygon):
        """Main method of the Transform, it takes as input the polygon to
        transform, and return the transformed polygon.

        Args:
            polygon (shapely.geometry.Geometry): Polygon to transform.

        Raises:
            NotImplementedError: Exception raised if this method is not
                overwritten by the subclass.
        """
        raise NotImplementedError


class NormalizePositionTransform(Transform):
    """Transform that normalize the position of the given polygon, so that the
    polygon is always is located in (0, 0), with positive coordinates.
    """

    def __call__(self, polygon):
        min_x, min_y, *_ = polygon.bounds
        return translate(polygon, -min_x, -min_y)


class NormalizeScaleTransform(Transform):
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


class NormalizeScaleStaticTransform(Transform):
    """Transform that normalize the scale of the given polygon, so that the
    polygon's size is always between 0 and 1. The scale is static across the
    dataset (does not change from one polygon to another).
    """

    def __init__(self, max_radius_range, *args, **kwargs):
        super().__init__()
        self.max_radius_range = max_radius_range

    def __call__(self, polygon):
        scale_size = self.max_radius_range * 4

        min_x, min_y, *_ = polygon.bounds
        ref = Point(min_x, min_y)
        scale_ratio = 1 / scale_size

        return scale(polygon, xfact=scale_ratio, yfact=scale_ratio, origin=ref)


TRANSFORMS = {
    "norm_pos": NormalizePositionTransform,
    "norm_scale": NormalizeScaleTransform,
    "norm_static_scale": NormalizeScaleStaticTransform
}
