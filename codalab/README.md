# Codalab

This directory contains the necessary components for the Codalab competition.

## Create the bundle

A Codalab competition is created from a `bundle`. You can create the bundle for the `cartesius` competition by following these steps.

* Build the solution file : `python codalab/build_solution.py`
* Build the sample submission file : `python codalab/build_solution.py out_file="submission.json" sample_submission=True`
* Zip the solution file : `zip codalab/compet_bundle/solution.zip solution.json`
* Zip the sample submission file : `zip sample_submission.zip submission.json`
* Zip the public files : `zip -j codalab/compet_bundle/files.zip sample_submission.zip cartesius/data/testset.json cartesius/data/valset.json`
* Zip the scoring script : `zip -j codalab/compet_bundle/scoring_program.zip codalab/scoring_program/*`
* Bundle everything : `zip -j codalab/compet_bundle.zip codalab/compet_bundle/*`
