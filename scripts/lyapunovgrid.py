# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# project
from ophiuchus.experiments import LyapunovGrid, ExperimentRunner

ExperimentRunner.parser.add_argument("--potential", dest="potential_name", required=True,
                                     help="Name of the potential YAML file.")

runner = ExperimentRunner(ExperimentClass=LyapunovGrid)
runner.run()

sys.exit(0)
