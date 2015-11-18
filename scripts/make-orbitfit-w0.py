# coding: utf-8

""" Make w0.npy files from orbitfit results

Call like:
TODO
python fitorbit.py --output-path=../output/orbitfits/ --potential=barred_mw \
-v --nsteps=256 --nwalkers=64 --mpi  --fixtime

"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import cPickle as pickle
import os

# Third-party
from astropy import log as logger
import numpy as np

# This project
from ophiuchus.data import OphiuchusData

def main(top_output_path, split_ix=256, potential_name=None, overwrite=False):
    all_ophdata = OphiuchusData()

    # top-level output path where orbitfit saved
    top_output_path = os.path.abspath(os.path.expanduser(top_output_path))
    output_path = os.path.join(top_output_path, "orbitfit")

    if potential_name is not None:
        paths = [potential_name]
    else:
        paths = os.listdir(output_path)

    for potential_name in paths:
        if potential_name.startswith("."): continue

        this_path = os.path.join(output_path, potential_name)
        w0_filename = os.path.join(this_path, "w0.npy")
        if os.path.exists(w0_filename) and overwrite:
            os.remove(w0_filename)

        if os.path.exists(w0_filename):
            logger.debug("File {} exists".format(w0_filename))
            continue

        with open(os.path.join(this_path, "sampler.pickle")) as f:
            sampler = pickle.load(f)

        _x0 = np.vstack(sampler.chain[:,split_ix:,:5])
        w0 = all_ophdata._mcmc_sample_to_w0(_x0.T).T

        # convert to w0 and save
        np.save(w0_filename, w0)

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

    parser.add_argument("--output-path", dest="output_path",
                        required=True, help="Path to save the output file.")
    parser.add_argument("--potential", dest="potential_name", default=None,
                        help="Name of the potential YAML file.")
    parser.add_argument("--ix", dest="ix", type=int, default=None,
                        help="Chain split.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.output_path, potential_name=args.potential_name,
         split_ix=args.ix, overwrite=args.overwrite)
