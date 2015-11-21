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
from ophiuchus.plot import plot_data_stream
from ophiuchus.mockstreamgrid import MockStreamGrid

def main(path, n):
    # load mock streams
    grid = MockStreamGrid.from_config(os.path.abspath(path), config_filename="mockstreamgrid.cfg")
    grid_d = grid.read_cache()[:n]
    streams = grid_d['w']
    nsuccess = grid_d['success'].sum()
    release_every = grid_d['release_every']
    dt = grid_d['dt']
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

        t = (np.arange(streams[i].shape[0])/2.).astype(int) / 1000. * release_every[i] * dt[i] # Gyr
        t = t - t.max()
        logger.debug("Plotting stream {}".format(i))
        fig = plot_data_stream(ophdata, stream_w=streams[i],
                               stream_t=t, stream_style=dict(s=7, alpha=0.75, cmap='plasma'))
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
    parser.add_argument("-n", dest="n", default=128, type=int,
                        help="Number to plot")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.path, args.n)
