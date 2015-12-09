# coding: utf-8

""" Class for running frequency mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
from six.moves import cPickle as pickle

# Third-party
import numpy as np
from astropy import log as logger

# Project
from streammorphology.experimentrunner import OrbitGridExperiment
from streammorphology.config import ConfigNamespace
from . import potential as op

class OphOrbitGridExperiment(OrbitGridExperiment):

    def __init__(self, cache_path, overwrite=False, **kwargs):

        # validate cache path
        self.cache_path = os.path.abspath(cache_path)
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)

        # create empty config namespace
        ns = ConfigNamespace()

        for k,v in self.config_defaults.items():
            if k not in kwargs:
                setattr(ns, k, v)
            else:
                setattr(ns, k, kwargs[k])

        self.config = ns

        self.cache_file = os.path.join(self.cache_path, self.config.cache_filename)
        if os.path.exists(self.cache_file) and overwrite:
            os.remove(self.cache_file)

        # load initial conditions
        if self.config.w0_path is None:
            w0_path = os.path.join(self.cache_path, self.config.w0_filename)
        else:
            w0_path = os.path.abspath(os.path.join(self.cache_path, self.config.w0_path))
            w0_path = os.path.join(w0_path, self.config.w0_filename)

        if not os.path.exists(w0_path):
            raise IOError("Initial conditions file '{0}' doesn't exist! You need"
                          "to generate this file first using make_grid.py".format(w0_path))
        self.w0 = np.load(w0_path)[:self.config.norbits]
        self.norbits = len(self.w0)
        logger.info("Number of orbits: {0}".format(self.norbits))

    def _run_wrapper(self, index):
        logger.info("Orbit {0}".format(index))

        # unpack input argument dictionary
        potname = self.config.potential_name
        potential = op.load_potential(potname)

        # read out just this initial condition
        norbits = len(self.w0)
        allfreqs = np.memmap(self.cache_file, mode='r',
                             shape=(norbits,), dtype=self.cache_dtype)

        # short-circuit if this orbit is already done
        if allfreqs['success'][index]:
            logger.debug("Orbit {0} already successfully completed.".format(index))
            return None

        # Only pass in things specified in _run_kwargs (w0 and potential required)
        kwargs = dict([(k,self.config[k]) for k in self.config.keys() if k in self._run_kwargs])
        res = self.run(w0=self.w0[index], potential=potential, **kwargs)
        res['index'] = index

        # cache res into a tempfile, return name of tempfile
        tmpfile = os.path.join(self._tmpdir, "{0}-{1}.pickle".format(self.__class__.__name__, index))
        with open(tmpfile, 'wb') as f:
            pickle.dump(res, f)
        return tmpfile
