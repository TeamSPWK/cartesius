class Task:
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


class GuessArea(Task):
    def get_label(self, polygon):
        return polygon.area
