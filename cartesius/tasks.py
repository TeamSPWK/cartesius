import torch.nn.functional as F
import numpy as np

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
            return 1
        else:
            return polygon.area / convex_p.area


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
    
    
class GuessOmbrRatio(Task):
    """Task predicting the OMBR(oriented minimum bounding rectangle) ratio of the polygon.
    """
    
    def get_label(self, polygon):
        return polygon.area / polygon.minimum_rotated_rectangle.area
    
    
class GuessAspectRatio(Task):
    """Task predicting the OMBR(oriented minimum bounding rectangle) aspect ratio of the polygon.
    """
    
    def get_label(self, polygon):
        ombr_coords = np.array(polygon.minimum_rotated_rectangle.exterior.coords)
        segvecs = ombr_coords[2] - ombr_coords[1], ombr_coords[1] - ombr_coords[0]
        x, y = [np.linalg.norm(vec) for vec in segvecs]
        return x / y if x < y else y / x
    
    
class GuessOpeningRatio(Task):
    """Task predicting the opening ratio of the polygon.
    Used fixed length of opening, to match the purpose of discriminating deadspace.
    #TODO: This task needs parameters from transformation (To be implemented)
    """
    
    def get_label(self, polygon):
        return polygon.buffer(-0.1).buffer(0.1).area / polygon.area
    
    
class GuessLongestThreeEdges(Task):
    """Task sorting the edges of the polygon by its lengths.
    #TODO: This task needs entity head and categorical loss (To be implemented)
    """
    
    def get_label(self, polygon):
        coords = np.array(polygon.exterior.coords)
        seglens = [np.linalg.norm(coords[i+1] - coords[i]) for i in range(len(coords) - 1)]
        return np.argsort(seglens[::-1])[:3]
    
    def get_head(self):
        return ScoreHead(self.d_in, self.dropout, 3)
    
    
        

TASKS = {
    "area": GuessArea,
    "perimeter": GuessPerimeter,
    "size": GuessSize,
    "concavity": GuessConcavity,
    "min_clear": GuessMinimumClearance,
    "centroid": GuessCentroid,
    "ombr_ratio": GuessOmbrRatio,
    "aspect_ratio": GuessAspectRatio,
    "opening_ratio": GuessOpeningRatio,
    "longest_edges": GuessLongestThreeEdges
}
