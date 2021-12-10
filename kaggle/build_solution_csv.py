import csv

from cartesius.data import PolygonTestset
from cartesius.tasks import TASKS
from cartesius.transforms import TRANSFORMS
from cartesius.utils import kaggle_convert_labels, load_conf


SOLUTION_FILE_NAME = "kaggle_solution.csv"


if __name__ == "__main__":
    conf = load_conf()
    tasks = [TASKS[t](conf) for t in conf.tasks]
    task_names = list(TASKS.keys())

    transforms = [TRANSFORMS["norm_scale"](conf), TRANSFORMS["norm_static_scale"](conf)]
    test_set = PolygonTestset(conf.test_set_file, tasks=tasks, transforms=transforms)

    kaggle_rows = [kaggle_convert_labels(task_names, labels, conf.tasks_scales) for _, labels in test_set]

    with open(SOLUTION_FILE_NAME, "w") as csv_f:
        fields = ["Usage"] + list(kaggle_rows[0][0].keys())
        writer = csv.DictWriter(csv_f, fieldnames=fields)

        writer.writeheader()

        for i, kaggle_row in enumerate(kaggle_rows):
            for row in kaggle_row:
                row["Usage"] = "Public"
                row["id"] = f"{i}_" + row["id"]
                writer.writerow(row)
