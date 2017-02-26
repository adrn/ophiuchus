# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from argparse import ArgumentParser
import logging
import os
from six.moves import cPickle as pickle

# Third-party
import numpy as np
from astropy import log as logger

# Project
from .. import RESULTSPATH
from .config import ConfigNamespace, save, load
from .. import potential as op

__all__ = ['ExperimentRunner']

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

    def parse_args(self):
        # Define parser object
        try:
            return self._args
        except AttributeError:
            self._args = self.parser.parse_args()
            return self._args

    def __init__(self, ExperimentClass):
        self.ExperimentClass = ExperimentClass

    def run(self, cache_path, **kwargs):
        args = self.parse_args()

        for k,v in kwargs.items():
            if hasattr(args, k):
                # overwrite with kwarg value
                setattr(args, k, v)

        if args.config_filename is None:
            raise ValueError("You must specify 'config_filename'")

        np.random.seed(args.seed)

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
        from gala.util import get_pool # TODO: moved to schwimmbad
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

        # extra kwargs
        extra_kwargs = dict()
        for k,v in vars(args).items():
            if k.startswith("_"):
                extra_kwargs[k.lstrip("_")] = v

        # Instantiate the experiment class
        with self.ExperimentClass.from_config(cache_path=cache_path,
                                              config_filename=args.config_filename,
                                              overwrite=args.overwrite,
                                              **extra_kwargs) as experiment:
            experiment._ensure_cache_exists()

            if index is None:
                indices = np.arange(experiment.ngrid, dtype=int)
            else:
                indices = np.arange(experiment.ngrid, dtype=int)[index]

            try:
                pool.map(experiment, indices, callback=experiment.callback)
            except:
                pool.close()
                logger.error("Unexpected error!")
                raise
            else:
                pool.close()
