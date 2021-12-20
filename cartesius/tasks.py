import torch.nn.functional as F

from cartesius.modeling import ScoreHead


class Task:
    """Base class for Tasks.

    When subclassing this class, you should define `get_label()` method, which compute
    the labels from the polygon for this task.
    You may optionally re-define `get_head()` and `get_loss_fn()`.

    Args:
        conf ([type]): [description]
    """

    def __init__(self, conf):
        self.d_in = conf["d_model"]
        self.dropout = conf["task_dropout"]

    def get_label(self, polygon):
        """Method used to retrieve the labels for this task, given the polygon used
        as input.

        Args:
            polygon (shapely.geometry.Geometry): Shapely geometry used as input.

        Raises:
            NotImplementedError: Exception raised if this method is not overwritten
                in a subclass.
        """
        raise NotImplementedError

    def get_head(self):
        """Method used to retrieve the task-specific classification/regression head.

        By default, it returns a head that compute a number.

        Returns:
            torch.nn.Module: Initialized classification/regression head.
        """
        return ScoreHead(self.d_in, self.dropout)

    def get_loss_fn(self):
        """Method used to retrieve the task-specific loss.

        By default, it returns the MSE loss.

        Returns:
            function: The loss function, that takes (preds, labels) as input and return
                the loss.
        """
        return F.mse_loss


class GuessArea(Task):
    """Task predicting the area of the polygon.
    """

    def get_label(self, polygon):
        return polygon.area


class GuessPerimeter(Task):
    """Task predicting the perimeter of the polygon.
    """

    def get_label(self, polygon):
        return polygon.length


class GuessSize(Task):
    """Task predicting the size (width + height) of the polygon.
    """

    def get_label(self, polygon):
        x_min, y_min, x_max, y_max = polygon.bounds
        return x_max - x_min, y_max - y_min

    def get_head(self):
        return ScoreHead(self.d_in, self.dropout, 2)


class GuessConcavity(Task):
    """Task predicting the concavity of the polygon.

    Concavity represents how much concave a polygon is. It's computed as the area
    of the current polygon divided by the area of its convex hull.
    """

    def get_label(self, polygon):
        convex_p = polygon.convex_hull

        if convex_p.area == 0:
            return 0.
        else:
            return max(polygon.area / convex_p.area, 0.)


class GuessMinimumClearance(Task):
    """Task predicting the minimum clearance of the polygon.

    The minimum clearance is the smallest distance by which a node could be moved
    to produce an invalid geometry.
    """

    def get_label(self, polygon):
        c = polygon.minimum_clearance
        if c < float("inf"):
            return c
        else:
            return 0


class GuessCentroid(Task):
    """Task predicting the centroid of the polygon.
    """

    def get_label(self, polygon):
        return polygon.centroid.coords[0]

    def get_head(self):
        return ScoreHead(self.d_in, self.dropout, 2)


TASKS = {
    "area": GuessArea,
    "perimeter": GuessPerimeter,
    "size": GuessSize,
    "concavity": GuessConcavity,
    "min_clear": GuessMinimumClearance,
    "centroid": GuessCentroid,
}
