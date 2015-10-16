# coding: utf-8

""" Class for running frequency mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import cPickle as pickle

# Third-party
import numpy as np
from astropy import log as logger

# Project
from streammorphology.experimentrunner import OrbitGridExperiment
from . import potential as op

class OphOrbitGridExperiment(OrbitGridExperiment):

    def _run_wrapper(self, index):
        logger.info("Orbit {0}".format(index))

        # unpack input argument dictionary
        import gary.potential as gp
        potential = gp.load(os.path.join(self.cache_path, self.config.potential_filename),
                            module=op)

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
        with open(tmpfile, 'w') as f:
            pickle.dump(res, f)
        return tmpfile
