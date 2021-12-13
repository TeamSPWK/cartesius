import csv
import sys

from omegaconf import OmegaConf as omg


def load_script_conf():
    default_conf = omg.create({
        "solution_file": "kaggle_solution.csv",
        "kaggle_submission_file": "submission.csv",
    })

    sys.argv = [a.strip("-") for a in sys.argv]
    cli_conf = omg.from_cli()

    return omg.merge(default_conf, cli_conf)


if __name__ == "__main__":
    conf = load_script_conf()
    with open(conf.solution_file, "r") as sol_f, open(conf.kaggle_submission_file, "r") as sub_f:
        solution_rows = csv.DictReader(sol_f)
        submission_rows = csv.DictReader(sub_f)

        first_row_read = False
        scores = []
        for x, y in zip(submission_rows, solution_rows):
            assert x["id"] == y["id"]

            if not first_row_read:
                first_row_read = True
                continue

            scores.append(abs(float(y["value"]) - float(x["value"])) * float(y["weight"]))

        print(sum(scores) / len(scores))
