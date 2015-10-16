# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# project
from streammorphology import ExperimentRunner
from ophiuchus.mockstreamgrid import MockStreamGrid

runner = ExperimentRunner(ExperimentClass=MockStreamGrid)
runner.run()

sys.exit(0)
