# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy import log as logger
import matplotlib.pyplot as pl
import numpy as np

# This project
from ophiuchus.data import OphiuchusData
from ophiuchus.plot import plot_data_orbit
from ophiuchus.mockstreamgrid import MockStreamGrid

def main(path):
    # load mock streams
    grid = MockStreamGrid.from_config(os.path.abspath(path), config_filename="mockstreamgrid.cfg")
    grid_d = grid.read_cache()
    streams = grid_d['w']
    nsuccess = grid_d['success'].sum()
    logger.info("{} successful".format(nsuccess))

    if nsuccess == 0:
        return

    # where to save the plots
    plot_path = os.path.join(path, "plots")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # load data
    ophdata = OphiuchusData()

    idx, = np.where(grid_d['success'])
    for i in idx:
        filename = os.path.join(plot_path, '{}.png'.format(i))
        if os.path.exists(filename):
            logger.debug("Stream {} plot exists".format(i))
            continue

        logger.debug("Plotting stream {}".format(i))
        fig = plot_data_orbit(ophdata, orbit_w=streams[i],
                              orbit_style=dict(marker='.', linestyle='none', alpha=0.1))
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
    parser.add_argument("--path", dest="path",
                        required=True, help="Path to the output file.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.path)
