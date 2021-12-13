#!/usr/bin/env python
import sys, os, os.path, json


WEIGTHS = {
    "area": 100,
    "perimeter": 1,
    "size": 50,
    "concavity": 20,
    "min_clear": 20,
    "centroid": 50,
}


def score(gold_data, submit_data):
    assert len(gold_data) == len(submit_data)
    scores = {k: [] for k in WEIGTHS.keys()}
    for gold_labels, preds in zip(gold_data, submit_data):
        for task in gold_labels.keys():
            if isinstance(gold_labels[task], list):
                row_scores = []
                for y, x in zip(gold_labels[task], preds[task]):
                    row_scores.append((y - x) ** 2)
                scores[task].append(sum(row_scores) / len(row_scores))
            else:
                scores[task].append((gold_labels[task] - preds[task]) ** 2)
    
    score = {k: sum(s) / len(s) for k, s in scores.items()}

    total_score = 0
    for task in WEIGTHS.keys():
        total_score += WEIGTHS[task] * score[task]

    print "Score : %s" % total_score
    return total_score

print "Starting..."

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res') 
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
	print "%s doesn't exist" % submit_dir

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')              
    output_file = open(output_filename, 'wb')

    gold_file = os.path.join(truth_dir, os.listdir(truth_dir)[0])
    submission_file = os.path.join(submit_dir, os.listdir(submit_dir)[0])

    print "Gold file : %s" % gold_file
    print "Submission file : %s" % submission_file

    with open(gold_file, "r") as f:
        gold_data = json.load(f)
    with open(submission_file, "r") as f:
        submit_data = json.load(f)

    output_file.write("MSE: %f" % score(gold_data, submit_data))
    output_file.close()
