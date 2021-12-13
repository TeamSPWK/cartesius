import json
import sys

from omegaconf import OmegaConf as omg

from cartesius.data import PolygonTestset
from cartesius.tasks import TASKS
from cartesius.transforms import TRANSFORMS
from cartesius.utils import kaggle_convert_labels, load_conf


def load_script_conf():
    default_conf = omg.create({
        "out_file": "solution.json",
        "sample_submission": False,
    })

    sys.argv = [a.strip("-") for a in sys.argv]
    cli_conf = omg.from_cli()

    return omg.merge(default_conf, cli_conf)


if __name__ == "__main__":
    conf = load_conf()
    script_conf = load_script_conf()
    tasks = [TASKS[t](conf) for t in conf.tasks]
    task_names = list(TASKS.keys())

    transforms = [TRANSFORMS["norm_pos"](conf), TRANSFORMS["norm_static_scale"](conf)]
    test_set = PolygonTestset(conf.test_set_file, tasks=tasks, transforms=transforms)

    rows = [{task_name: list(label) if isinstance(label, tuple) else label for task_name, label in zip(task_names, labels)} for _, labels in test_set]

    if script_conf.sample_submission:
        rows = [{k: [0 for _ in v] if isinstance(v, list) else 0 for k, v in row.items()} for row in rows]

    with open(script_conf.out_file, "w") as f:
        json.dump(rows, f)
