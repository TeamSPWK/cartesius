import pathlib
import sys

import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from omegaconf import OmegaConf as omg


CONFIG_DIR = "config"


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


def load_yaml(yaml_file):
    """Function to load a configuration file, and recursively load parent configuration
    files.

    Note:
        This funciton is recursive.

    Args:
        yaml_file (str): Path of the config file to load.

    Returns:
        omegaconf.OmegaConf: Loaded configuration.
    """
    # Try to load the file from the local config directory
    try:
        conf_dir_path = pathlib.Path(__file__).parent.resolve() / CONFIG_DIR / yaml_file
        conf = omg.load(conf_dir_path)
    except FileNotFoundError:
        # Then maybe the user provided a path from the working dir ? Try to load it directly
        conf = omg.load(yaml_file)

    if conf.parent_config:
        parent_conf = load_yaml(conf.parent_config)
        conf = omg.merge(parent_conf, conf)

    return conf


def load_conf():
    """Function loading the configuration.

    This function will look for a configuration file path given from command
    line, and if not given, will use a default configuration file path.
    It will then load the configuration, retrieve parent configuration if any,
    and merge it with the command line arguments.

    Returns:
        omegaconf.OmegaConf: Loaded configuration.
    """
    default_conf = omg.create({"config": "default.yaml"})

    sys.argv = [a.strip("-") for a in sys.argv]
    cli_conf = omg.from_cli()

    yaml_file = omg.merge(default_conf, cli_conf).config

    yaml_conf = load_yaml(yaml_file)

    return omg.merge(default_conf, yaml_conf, cli_conf)
