import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon


def print_polygon(poly, *args, **kwargs):
    """Function to print a shapely Geometry using matplotlib `plt.plot()` function.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from cartesius.utils import print_polygon

        >>> plt.clf()
        >>> print_polygon(poly1)
        >>> print_polygon(poly2, color="tab:blue", linestyle="-")
        >>> plt.gca().set_aspect(1)
        >>> plt.axis("off")
        >>> plt.savefig("whatever.png")

    Args:
        poly (shapely.geometry.Geometry): Shapely Geometry to print.
        *args: Positional arguments to pass to the `plt.plot()` function.
        **kwargs: keyword arguments to pass to the `plt.plot()` function. Note that if
            kwargs contains `fill` key, the `plt.fill_between()` function is used
            instead, and the `facecolor` argument is set to the value of the `fill` key.
    """
    if poly.is_empty:
        return

    if isinstance(poly, Point):
        return
    if isinstance(poly, LineString):
        xy = poly.xy
    elif isinstance(poly, Polygon):
        xy = poly.exterior.xy
    else:
        for pol in poly:
            print_polygon(pol, *args, **kwargs)
        return

    fill = kwargs.pop("fill", False)
    if fill:
        plt_fn = plt.fill_between
        kwargs["facecolor"] = fill
    else:
        plt_fn = plt.plot

    plt_fn(*xy, *args, **kwargs)


def save_polygon(*polygons, path="poly.png"):
    """Function to save an image containing the given polygons.

    Example:
        >>> from cartesius.utils import save_polygon
        >>> save_polygon(poly1, poly2)

    Args:
        polygons (shapely.geometry.Geometry): Shapely Geometry to print.
        path (str): Path where to save the image.
    """
    plt.clf()
    for p in polygons:
        print_polygon(p)
    plt.gca().set_aspect(1)
    plt.axis("off")
    plt.savefig(path)
