import cv2
import numpy as np
from shapely.affinity import scale
from shapely.affinity import translate
from shapely.geometry import Point


class Transform:
    """Base class for transforms. A Transform is a callable that take a polygon
    as input and transform it appropriately.

    Args:
        config (omegaconf.OmegaConf): Configuration.
    """

    def __init__(self, config):
        self.config = config

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

    def __call__(self, polygon):
        scale_size = max(self.config["avg_radius_range"]) * 4

        min_x, min_y, *_ = polygon.bounds
        ref = Point(min_x, min_y)
        scale_ratio = 1 / scale_size

        return scale(polygon, xfact=scale_ratio, yfact=scale_ratio, origin=ref)


class RasterTransform(Transform):
    """Transform that rasterize the given polygon
    """

    def __call__(self, polygon):
        canvas_len = self.config["canvas_len"]
        poly_max_len = self.config["poly_max_len"]
        canvas_center = np.array((canvas_len, canvas_len)) / 2
        poly_pts = np.array(polygon.boundary)[:-1]
        centralized_poly = ((poly_pts - np.array(polygon.centroid)) / (poly_max_len / canvas_len) + canvas_center)
        preprocessed_poly = centralized_poly.astype(np.int32).reshape(-1, 1, 2)
        canvas = np.zeros((canvas_len, canvas_len, 3), dtype=np.uint8)
        img = cv2.fillPoly(canvas, [preprocessed_poly.reshape(-1, 1, 2)], (255, 255, 255))

        return img, centralized_poly


TRANSFORMS = {
    "norm_pos": NormalizePositionTransform,
    "norm_scale": NormalizeScaleTransform,
    "norm_static_scale": NormalizeScaleStaticTransform
}
