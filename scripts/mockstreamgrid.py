# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# project
from ophiuchus import RESULTSPATH
from ophiuchus.experiments import MockStreamGrid, ExperimentRunner

ExperimentRunner.parser.add_argument("--potential", dest="_potential_name", required=True,
                                     help="Name of the potential YAML file.")

runner = ExperimentRunner(ExperimentClass=MockStreamGrid)

# Custom cache path
args = runner.parse_args()
if args.results_path is None:
    results_path = RESULTSPATH
else:
    results_path = os.path.abspath(os.path.expanduser(args.results_path))

if results_path is None:
    raise ValueError("If $PROJECTSPATH is not set, you must provide a path to save "
                     "the results in with the --results_path argument.")

experiment_name = runner.ExperimentClass.__name__.lower().rstrip("grid")
cache_path = os.path.join(results_path, args._potential_name, experiment_name)

runner.run(cache_path=cache_path)

sys.exit(0)
