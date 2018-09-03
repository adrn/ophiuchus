# coding: utf-8

""" Compute Lyapunov exponents for the mean orbit fits """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy import log as logger
import numpy as np
import gala.dynamics as gd
from six.moves import cPickle as pickle

# This project
from ophiuchus import RESULTSPATH
import ophiuchus.potential as op

def main(potential_name, results_path=None, overwrite=False):
    # top-level output path for saving (this will create a subdir within output_path)
    if results_path is None:
        top_path = RESULTSPATH
    else:
        top_path = os.path.abspath(os.path.expanduser(results_path))

    if top_path is None:
        raise ValueError("If $PROJECTSPATH is not set, you must provide a path to save "
                         "the results in with the --results_path argument.")

    output_path = os.path.join(top_path, potential_name, "lyapunov")
    logger.debug("Output path: {}".format(output_path))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    T = 200. # The orbits have periods ~200 Myr
    nsteps_per_period = 1024
    nperiods = 1024
    dt = T / nsteps_per_period
    nsteps = int(nperiods * nsteps_per_period)

    output_file = os.path.join(output_path, 'lyap.pickle')
    if os.path.exists(output_file) and overwrite:
        os.remove(output_file)

    if os.path.exists(output_file):
        logger.info("Output file exists. Exiting.")
        return

    # load initial conditions file
    w0_path = os.path.join(top_path, potential_name, 'orbitfit', 'w0.npy')
    w0 = np.ascontiguousarray(np.load(w0_path)[0]) # load only the mean orbit
    pot = op.load_potential(potential_name)

    # compute lyapunov exponent
    lyap,orbit = gd.fast_lyapunov_max(w0, pot, dt=dt, nsteps=nsteps)
    with open(output_file, 'wb') as f:
        pickle.dump(lyap, f)
    # np.save(output_file, lyap.decompose(pot.units).value)

    # orbit_output_file = os.path.join(output_path, 'orbit.npy')
    orbit_output_file = os.path.join(output_path, 'orbit.pickle')
    with open(orbit_output_file, 'wb') as f:
        pickle.dump(orbit, f)
    # np.save(orbit_output_file, (orbit.t.decompose(pot.units).value,
    #                             orbit.pos.decompose(pot.units).value,
    #                             orbit.vel.decompose(pot.units).value))


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true", help="Overwrite any existing data.")

    parser.add_argument("--results-path", dest="results_path", default=None,
                        help="Path to save the output file.")
    parser.add_argument("--potential", dest="potential_name", required=True,
                        help="Name of the potential YAML file.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.potential_name, results_path=args.results_path, overwrite=args.overwrite)
