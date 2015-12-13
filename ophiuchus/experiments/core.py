# coding: utf-8

from __future__ import division, print_function

""" Note: this is basically the same file as streammorphology/experimentrunner.py """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from abc import ABCMeta, abstractproperty, abstractmethod

from argparse import ArgumentParser
import logging
import os
from six.moves import cPickle as pickle

# Third-party
import numpy as np
from astropy import log as logger
from gary.util import get_pool

# Project
from .. import RESULTSPATH
from .config import ConfigNamespace, save, load
from .. import potential as op

__all__ = ['GridExperiment']

class GridExperiment(object):

    __metaclass__ = ABCMeta

    config_defaults = dict()

    def __init__(self, cache_path, overwrite=False, **kwargs):

        # validate cache path
        self.cache_path = os.path.abspath(cache_path)
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)

        # create empty config namespace
        ns = ConfigNamespace()

        for k in self.required_kwargs:
            if k not in kwargs:
                raise ValueError("'{}' is a required keyword argument".format(k))

        for k,v in self.config_defaults.items():
            if k not in kwargs:
                setattr(ns, k, v)
            else:
                setattr(ns, k, kwargs[k])

        for k,v in kwargs.items():
            if not hasattr(ns, k):
                setattr(ns, k, kwargs[k])

        self.config = ns

        self.cache_file = os.path.join(self.cache_path, self.config.cache_filename)
        if os.path.exists(self.cache_file) and overwrite:
            os.remove(self.cache_file)

        # the length of the grid or the number of iterations to make
        self.ngrid = self.config.get('ngrid', len(self.grid))

    # Context management
    def __enter__(self):
        self._tmpdir = os.path.join(self.cache_path, "_tmp_{0}".format(self.__class__.__name__))
        logger.debug("Creating temp. directory {0}".format(self._tmpdir))
        if os.path.exists(self._tmpdir):
            import shutil
            shutil.rmtree(self._tmpdir)
        os.mkdir(self._tmpdir)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if os.path.exists(self._tmpdir):
            logger.debug("Removing temp. directory {0}".format(self._tmpdir))
            import shutil
            shutil.rmtree(self._tmpdir)

    def _ensure_cache_exists(self):
        # make sure memmap file exists
        if not os.path.exists(self.cache_file):
            # make sure memmap file exists
            d = np.memmap(self.cache_file, mode='w+',
                          dtype=self.cache_dtype, shape=(self.ngrid,))
            d[:] = np.zeros(shape=(self.ngrid,), dtype=self.cache_dtype)

    def read_cache(self):
        """
        Read the numpy memmap'd file containing cached results from running
        this experiment. This function returns a numpy structured array
        with named columns and proper data types.
        """

        # first get the memmap array
        return np.memmap(self.cache_file, mode='r', shape=(self.ngrid,),
                         dtype=self.cache_dtype)

    def dump_config(self, config_filename):
        """
        Write the current configuration out to the specified filename.
        """
        save(self.config, config_filename)

    @classmethod
    def from_config(cls, cache_path, config_filename, overwrite=False, **kwargs):
        """
        Read the state from a configuration file.
        """
        if not os.path.exists(config_filename):
            config_path = os.path.abspath(os.path.join(cache_path, config_filename))
        else:
            config_path = config_filename

        d = load(config_path)
        for k,v in d.items():
            if k not in kwargs:
                kwargs[k] = v

        return cls(cache_path=cache_path, overwrite=overwrite, **kwargs)

    def callback(self, tmpfile):
        """
        TODO:
        """

        if tmpfile is None:
            logger.debug("Tempfile is None")
            return

        with open(tmpfile,'rb') as f:
            result = pickle.load(f)
        os.remove(tmpfile)

        logger.debug("Flushing {0} to output array...".format(result['index']))
        memmap = np.memmap(self.cache_file, mode='r+',
                           dtype=self.cache_dtype, shape=(seld.ngrid,))
        if result['error_code'] != 0.:
            logger.error("Error code = {0}".format(result['error_code']))
            # error happened
            for key in memmap.dtype.names:
                if key in result:
                    memmap[key][result['index']] = result[key]

        else:
            # all is well
            for key in memmap.dtype.names:
                memmap[key][result['index']] = result[key]

        # flush to output array
        memmap.flush()
        logger.debug("...flushed, washing hands.")

        del result
        del memmap

    def __call__(self, index):
        return self._run_wrapper(index)

    def _run_wrapper(self, index):
        logger.info("Index {0}".format(index))

        # short-circuit if this index is already done
        d = np.memmap(self.cache_file, mode='r', shape=(self.ngrid,), dtype=self.cache_dtype)
        if d['success'][index]:
            logger.debug("Index {0} already successfully completed.".format(index))
            return None

        # get results from this index
        res = self.run(index=index)
        res['index'] = index

        # cache res into a tempfile, return name of tempfile
        tmpfile = os.path.join(self._tmpdir, "{0}-{1}.pickle".format(self.__class__.__name__, index))
        with open(tmpfile, 'wb') as f:
            pickle.dump(res, f)
        return tmpfile

    def status(self):
        """
        Prints out (to the logger) the status of the current run of the experiment.
        """

        d = self.read_cache()

        # numbers
        nsuccess = d['success'].sum()
        nfail = ((d['success'] is False) & (d['error_code'] > 0)).sum()

        # TODO: why don't logger.info() calls work here??
        # logger.info("------------- {0} Status -------------".format(self.__class__.__name__))
        # logger.info("Total number of orbits: {0}".format(len(d)))
        # logger.info("Successful: {0}".format(nsuccess))
        # logger.info("Failures: {0}".format(nfail))

        # for ecode in sorted(self.error_codes.keys()):
        #     nfail = (d['error_code'] == ecode).sum()
        #     logger.info("\t({0}) {1}: {2}".format(ecode, self.error_codes[ecode], nfail))

        print("------------- {0} Status -------------".format(self.__class__.__name__))
        print("Total number of orbits: {0}".format(len(d)))
        print("Successful: {0}".format(nsuccess))
        print("Failures: {0}".format(nfail))

        for ecode in sorted(self.error_codes.keys()):
            nfail = (d['error_code'] == ecode).sum()
            print("\t({0}) {1}: {2}".format(ecode, self.error_codes[ecode], nfail))

    # ------------------------------------------------------------------------
    # Subclasses must implement:

    @abstractproperty
    def error_codes(self):
        """ A dict mapping from integer error code to string describing the error """

    @abstractproperty
    def cache_dtype(self):
        """ The (numpy) dtype of the memmap'd cache file """

    @abstractproperty
    def required_kwargs(self):
        """ A list of required names of arguments to run() """

    @abstractmethod
    def run(self, index):
        """ (classmethod) Run the experiment on a single orbit """

    @abstractproperty
    def grid(self):
        """ """
