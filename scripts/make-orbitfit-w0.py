# coding: utf-8

""" Make w0.npy files from orbitfit results """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import acor
from astropy import log as logger
import numpy as np
from six.moves import cPickle as pickle

# This project
from ophiuchus import RESULTSPATH
from ophiuchus.data import OphiuchusData
from ophiuchus.util import integrate_forward_backward
from ophiuchus.plot import plot_data_orbit
import ophiuchus.potential as op

def main(potential_name, results_path=None, split_ix=None, overwrite=False):
    all_ophdata = OphiuchusData()

    # top-level output path for saving (this will create a subdir within output_path)
    if results_path is None:
        top_path = RESULTSPATH
    else:
        top_path = os.path.abspath(os.path.expanduser(results_path))

    if top_path is None:
        raise ValueError("If $PROJECTSPATH is not set, you must provide a path to save "
                         "the results in with the --results_path argument.")

    output_path = os.path.join(top_path, potential_name, "orbitfit")
    logger.debug("Output path: {}".format(output_path))

    w0_filename = os.path.join(output_path, "w0.npy")
    if os.path.exists(w0_filename) and overwrite:
        os.remove(w0_filename)

    if os.path.exists(w0_filename):
        logger.debug("File {} exists".format(w0_filename))
        continue

    with open(os.path.join(output_path, "sampler.pickle"), 'rb') as f:
        sampler = pickle.load(f)

    # default is to split in half
    if split_ix is None:
        split_ix = sampler.chain.shape[1] // 2

    # measure the autocorrelation time for each parameter
    taus = []
    for i in range(sampler.chain.shape[-1]):
        tau,_,_ = acor.acor(sampler.chain[:,split_ix:,i])
        taus.append(tau)
    logger.debug("Autocorrelation times: {}".format(taus))
    every = int(2*max(taus)) # take every XX step

    _x0 = np.vstack(sampler.chain[:,split_ix::every,:5])
    np.random.shuffle(_x0)
    w0 = all_ophdata._mcmc_sample_to_w0(_x0.T).T

    mean_w0 = all_ophdata._mcmc_sample_to_w0(np.mean(_x0, axis=0)).T
    w0 = np.vstack((mean_w0, w0))

    logger.info("{} initial conditions after thinning chains".format(w0.shape[0]))

    # convert to w0 and save
    np.save(w0_filename, w0)

    potential = op.load_potential(potential_name)

    # plot orbit fits
    ix = np.random.randint(len(sampler.flatchain), size=64)
    fig = plot_data_orbit(all_ophdata)
    for sample in sampler.flatchain[ix]:
        sample_w0 = all_ophdata._mcmc_sample_to_w0(sample[:5])[:,0]
        tf,tb = (5.,-5.)
        w = integrate_forward_backward(potential, sample_w0, t_forw=tf, t_back=tb)
        fig = plot_data_orbit(all_ophdata, orbit_w=w, data_style=dict(marker=None),
                              orbit_style=dict(color='#2166AC', alpha=0.1), fig=fig)
    fig.savefig(os.path.join(output_path, "orbits.png"), dpi=300)

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

    parser.add_argument("--results-path", dest="results_path",
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

    main(potential_name=args.potential_name, results_path=args.results_path,
         split_ix=args.ix, overwrite=args.overwrite)
