import csv


SOLUTION_FILE_NAME = "kaggle_solution.csv"
SUBMISSION_FILE_NAME = "submission.csv"


if __name__ == "__main__":
    with open(SOLUTION_FILE_NAME, "r") as sol_f, open(SUBMISSION_FILE_NAME, "r") as sub_f:
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
