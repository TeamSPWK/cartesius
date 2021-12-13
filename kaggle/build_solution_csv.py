import csv
import sys

from omegaconf import OmegaConf as omg

from cartesius.data import PolygonTestset
from cartesius.tasks import TASKS
from cartesius.transforms import TRANSFORMS
from cartesius.utils import kaggle_convert_labels, load_conf


def load_script_conf():
    default_conf = omg.create({
        "solution_file": "kaggle_solution.csv",
    })

    conf = load_conf()
    sys.argv = [a.strip("-") for a in sys.argv]
    cli_conf = omg.from_cli()

    return omg.merge(default_conf, conf, cli_conf)


if __name__ == "__main__":
    conf = load_script_conf()
    tasks = [TASKS[t](conf) for t in conf.tasks]
    task_names = list(TASKS.keys())

    transforms = [TRANSFORMS["norm_pos"](conf), TRANSFORMS["norm_static_scale"](conf)]
    test_set = PolygonTestset(conf.test_set_file, tasks=tasks, transforms=transforms)

    kaggle_rows = [kaggle_convert_labels(task_names, labels, conf.tasks_scales) for _, labels in test_set]

    with open(conf.solution_file, "w") as csv_f:
        fields = ["Usage"] + list(kaggle_rows[0][0].keys())
        writer = csv.DictWriter(csv_f, fieldnames=fields)

        writer.writeheader()

        for i, kaggle_row in enumerate(kaggle_rows):
            for row in kaggle_row:
                row["Usage"] = "Public"
                row["id"] = f"{i}_" + row["id"]
                writer.writerow(row)
