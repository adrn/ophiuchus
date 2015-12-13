# coding: utf-8

from __future__ import division, print_function

""" Note: this is basically the same file as streammorphology/experimentrunner.py """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from abc import ABCMeta, abstractproperty
try:
    from abc import abstractclassmethod
except ImportError: # only works in Python 3
    class abstractclassmethod(classmethod):

        __isabstractmethod__ = True

        def __init__(self, callable):
            callable.__isabstractmethod__ = True
            super(abstractclassmethod, self).__init__(callable)

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

__all__ = ['OrbitGridExperiment', 'ExperimentRunner']

class OrbitGridExperiment(object):

    __metaclass__ = ABCMeta

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
                          dtype=self.cache_dtype, shape=(self.norbits,))
            d[:] = np.zeros(shape=(self.norbits,), dtype=self.cache_dtype)

    def read_cache(self):
        """
        Read the numpy memmap'd file containing cached results from running
        this experiment. This function returns a numpy structured array
        with named columns and proper data types.
        """

        # first get the memmap array
        return np.memmap(self.cache_file, mode='r', shape=(len(self.w0),),
                         dtype=self.cache_dtype)

    def dump_config(self, config_filename):
        """
        Write the current configuration out to the specified filename.
        """
        save(self.config, config_filename)

    @classmethod
    def from_config(cls, cache_path, config_filename, potential_name, overwrite=False):
        """
        Read the state from a configuration file.
        """
        if not os.path.exists(config_filename):
            config_path = os.path.abspath(os.path.join(cache_path, config_filename))
        else:
            config_path = config_filename
        return cls(cache_path=cache_path, overwrite=overwrite, potential_name=potential_name,
                   **load(config_path))

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
                           dtype=self.cache_dtype, shape=(len(self.w0),))
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
        logger.info("Orbit {0}".format(index))

        # unpack input argument dictionary
        potname = self.config.potential_name
        potential = op.load_potential(potname) # HACK: this is different from streammorphology

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
    def _run_kwargs(self):
        """ A list of the names of the keyword arguments used in `run()` (below) """

    @abstractproperty
    def config_defaults(self):
        """ A dict of configuration defaults """

    @abstractclassmethod
    def run(cls, w0, potential, **kwargs):
        """ (classmethod) Run the experiment on a single orbit """

# ----------------------------------------------------------------------------

class ExperimentRunner(object):

    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY. DESTROY. (default = False)")
    parser.add_argument("--seed", dest="seed", default=42, type=int,
                        help="Seed for random number generators.")

    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Use an MPI pool.")
    parser.add_argument("--results_path", dest="results_path", type=str, default=None,
                        help="Top level path to cache everything")
    parser.add_argument("-c", "--config", dest="config_filename", type=str, default=None,
                        help="Name of the config file (relative to the path).")

    parser.add_argument("--index", dest="index", type=str, default=None,
                        help="Specify a subset of orbits to run, e.g., "
                             "--index=20:40 to do only orbits 20-39.")

    def _parse_args(self):
        # Define parser object
        return self.parser.parse_args()

    def __init__(self, ExperimentClass):
        self.ExperimentClass = ExperimentClass

    def run(self, **kwargs):
        args = self._parse_args()

        for k,v in kwargs.items():
            if hasattr(args, k):
                # overwrite with kwarg value
                setattr(args, k, v)

        np.random.seed(args.seed)

        # top-level output path for saving (this will create a subdir within output_path)
        if args.results_path is None:
            results_path = RESULTSPATH
        else:
            results_path = os.path.abspath(os.path.expanduser(args.results_path))

        if results_path is None:
            raise ValueError("If $PROJECTSPATH is not set, you must provide a path to save "
                             "the results in with the --results_path argument.")
        experiment_name = self.__class__.__name__.lower().rstrip("grid")
        cache_path = os.path.join(results_path, args.potential_name, experiment_name)

        if args.config_filename is None:
            raise ValueError("You must define 'config_filename.'")

        # Set logger level based on verbose flags
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        elif args.quiet:
            logger.setLevel(logging.ERROR)
        else:
            logger.setLevel(logging.INFO)

        # if MPI, use load balancing
        if args.mpi:
            kwargs = dict(loadbalance=True)
        else:
            kwargs = dict()

        # get a pool object for multiprocessing / MPI
        pool = get_pool(mpi=args.mpi, **kwargs)
        if args.mpi:
            logger.info("|----------- Using MPI -----------|")
        else:
            logger.info("|----------- Running in serial -----------|")

        if args.index is None:
            index = None
        else:
            try:
                index = slice(*map(int, args.index.split(":")))
            except:
                try:
                    index = np.array(map(int,args.index.split(",")))
                except:
                    index = None

        # Instantiate the experiment class
        with self.ExperimentClass.from_config(cache_path=cache_path,
                                              config_filename=args.config_filename,
                                              overwrite=args.overwrite,
                                              potential_name=args.potential_name) as experiment:
            experiment._ensure_cache_exists()

            if index is None:
                indices = np.arange(experiment.norbits, dtype=int)
            else:
                indices = np.arange(experiment.norbits, dtype=int)[index]

            try:
                pool.map(experiment, indices, callback=experiment.callback)
            except:
                pool.close()
                logger.error("Unexpected error!")
                raise
            else:
                pool.close()
