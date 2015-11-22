# coding: utf-8

""" Compute Lyapunov exponents for the mean orbit fits """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import cPickle as pickle

# Third-party
from astropy import log as logger
import numpy as np
import gary.dynamics as gd

# This project
import ophiuchus.potential as op

def main(path, overwrite=False):
    dt = 1.
    nsteps = 2560000 # 256 steps per orbit -- 10000 orbital periods

    path = os.path.abspath(path)
    name = os.path.basename(path)
    output_file = os.path.join(path, 'lyap.npy')
    t_output_file = os.path.join(path, 'lyap_t.npy')
    ws_output_file = os.path.join(path, 'lyap_ws.npy')

    if os.path.exists(output_file) and overwrite:
        os.remove(output_file)

    if os.path.exists(output_file):
        logger.info("Output file exists. Exiting.")

    # load initial conditions file
    w0 = np.load(os.path.join(path, 'w0.npy'))
    w0 = np.median(w0, axis=0)
    pot = op.load_potential(name)

    # compute lyapunov exponent
    lyap,t,ws = gd.fast_lyapunov_max(w0, pot, dt=dt, nsteps=nsteps)
    np.save(output_file, lyap)
    np.save(t_output_file, t)
    np.save(ws_output_file, ws)

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

    parser.add_argument("--path", dest="path",
                        required=True, help="Path to the output file.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.path, overwrite=args.overwrite)
