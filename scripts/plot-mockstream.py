# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import astropy.units as u
from astropy import log as logger
import matplotlib.pyplot as pl
import numpy as np
from gary.dynamics import CartesianPhaseSpacePosition as CPSP

# This project
from ophiuchus import RESULTSPATH
import ophiuchus.potential as op
from ophiuchus.data import OphiuchusData
from ophiuchus.plot import plot_data_stream
from ophiuchus.experiments import MockStreamGrid

def main(potential_name, n, config_filename, results_path=None, overwrite=False):
    # top-level output path for saving (this will create a subdir within output_path)
    if results_path is None:
        top_path = RESULTSPATH
    else:
        top_path = os.path.abspath(os.path.expanduser(results_path))

    if top_path is None:
        raise ValueError("If $PROJECTSPATH is not set, you must provide a path to save "
                         "the results in with the --results_path argument.")

    output_path = os.path.join(top_path, potential_name, "mockstream")
    logger.debug("Reading path: {}".format(output_path))

    # load mock streams
    grid = MockStreamGrid.from_config(output_path, config_filename=config_filename,
                                      potential_name=potential_name)
    grid_d = grid.read_cache()[:n]
    streams = grid_d['w']
    nsuccess = grid_d['success'].sum()
    release_every = grid_d['release_every']
    dt = grid_d['dt']
    logger.info("{} successful".format(nsuccess))

    if nsuccess == 0:
        return

    pot = op.load_potential(potential_name)

    # where to save the plots
    plot_path = os.path.join(output_path, "plots")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # load data
    ophdata = OphiuchusData()

    idx, = np.where(grid_d['success'])
    for i in idx:
        filename = os.path.join(plot_path, 'colored_{}.png'.format(i))
        if os.path.exists(filename) and not overwrite:
            logger.debug("Stream {} plot exists".format(i))
            continue

        t = (np.arange(streams[i].shape[0])/2.).astype(int) / 1000. * release_every[i] * dt[i] # Gyr
        t = t - t.max()
        logger.debug("Plotting stream {}".format(i))

        stream = CPSP.from_w(streams[i].T, units=pot.units)
        fig = plot_data_stream(ophdata, stream=stream,
                               stream_style=dict(s=7, c=t, alpha=0.75, cmap='plasma_r'))
        fig.savefig(filename, dpi=300)
        pl.close(fig)

        # also plot non-colored
        filename = os.path.join(plot_path, 'transp_{}.png'.format(i))
        fig = plot_data_stream(ophdata, stream=stream,
                               stream_style=dict(s=3, color='#555555', alpha=0.15))
        fig.savefig(filename, dpi=300)
        pl.close(fig)

        # also plot a zoom in
        filename = os.path.join(plot_path, 'zoom_{}.png'.format(i))
        zoom_lims = {
            'l': [6.,3.7]*u.deg,
            'b': [30.5,32.2]*u.deg,
            'distance': [7.7,9.1]*u.kpc,
            'vr': [280,300]*u.km/u.s
        }
        fig = plot_data_stream(ophdata, stream=stream, lims=zoom_lims,
                               stream_style=dict(s=3, color='#555555', alpha=0.15))
        fig.savefig(filename, dpi=300)
        pl.close(fig)

        # also plot xyz
        filename = os.path.join(plot_path, 'xyz_{}.png'.format(i))
        fig = stream.plot(color='#555555', alpha=0.15, subplots_kwargs=dict(sharex=True, sharey=True))
        fig.axes[0].set_xlim(-15,15)
        fig.axes[0].set_ylim(-15,15)
        fig.savefig(filename, dpi=300)
        pl.close(fig)

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

    parser.add_argument("--potential", dest="potential_name", required=True,
                        help="Name of the potential YAML file.")
    parser.add_argument("--results_path", dest="results_path", type=str, default=None,
                        help="Top level path to cache everything")
    parser.add_argument("-c", "--config", dest="config_filename", type=str, required=True,
                        help="Name of the config file (relative to the path).")
    parser.add_argument("-n", dest="n", default=128, type=int,
                        help="Number to plot")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(potential_name=args.potential_name, config_filename=args.config_filename,
         n=args.n, results_path=args.results_path, overwrite=args.overwrite)
