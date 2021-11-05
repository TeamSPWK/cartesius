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
        self.d_hid = conf["task_d_ff"]
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
        return ScoreHead(self.d_in, self.d_hid, self.dropout)

    def get_loss_fn(self):
        """Method used to retrieve the task-specific loss.

        By default, it returns the MSE loss.

        Returns:
            function: The loss function, that takes (preds, labels) as input and return
                the loss.
        """
        return F.mse_loss


class GuessArea(Task):
    def get_label(self, polygon):
        return polygon.area


class GuessPerimeter(Task):
    def get_label(self, polygon):
        return polygon.length


TASKS = {
    "area": GuessArea,
    "perimeter": GuessPerimeter,
}
