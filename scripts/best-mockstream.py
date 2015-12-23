# coding: utf-8

""" From the mockstream runs, find the best mockstream that fits the data """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np

# Custom
import gary.coordinates as gc
import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp
from gary.observation import distance_modulus
from gary.units import galactic
from scipy.misc import logsumexp

import ophiuchus.potential as op
from ophiuchus.data import OphiuchusData
from ophiuchus.util import integrate_forward_backward
from ophiuchus.coordinates import Ophiuchus
from ophiuchus import galactocentric_frame, vcirc, vlsr, RESULTSPATH
from ophiuchus.experiments import MockStreamGrid
from ophiuchus.plot import plot_data_stream, surface_density

def ln_likelihood(model, ophdata):
    """
    Compute the likelihood by approximating the model density using a
    KDE of the points.

    Parameters
    ----------
    model : :class:`gary.dynamics.CartesianPhaseSpacePosition`
    ophdata : `ophiuchus.data.OphiuchusData`
    """

    # sum over model points, product over data points
    gal,vel = model.to_frame(coord.Galactic, galactocentric_frame=galactocentric_frame,
                             vcirc=vcirc, vlsr=vlsr)

    # extra dispersions from intrinsic thickness of the stream -- this comes from the
    #   orbit fitting sampler file
    phi2_sigma = 0.01 * u.deg # 0.00353495
    d_sigma = 0.3093834 * u.kpc
    vr_sigma = 0.00265714 * u.kpc/u.Myr # ~2 km/s

    # get model coordinates
    model_l, model_b, model_d = gal.l.radian, gal.b.radian, gal.distance.to(u.kpc).value
    model_vr = vel[2].decompose(galactic).value

    # get data coordinates
    data_l, data_b, data_d = ophdata.coord.l.radian,ophdata.coord.b.radian,ophdata.coord.distance.to(u.kpc).value
    data_vr = ophdata.veloc['vr'].decompose(galactic).value

    # variances
    var_l = var_b = np.atleast_1d(phi2_sigma.decompose(galactic).value)**2
    var_d = (ophdata.coord_err['distance']**2 + d_sigma**2).decompose(galactic).value # extra distance spread
    var_vr = (ophdata.veloc_err['vr']**2 + vr_sigma**2).decompose(galactic).value # extra velocity spread

    chi2 = -0.5*(
        ((data_l[None] - model_l[:,None])**2 / var_l[None] + np.log(2*np.pi*var_l[None])) +
        ((data_b[None] - model_b[:,None])**2 / var_b[None] + np.log(2*np.pi*var_b[None])) +
        ((data_d[None] - model_d[:,None])**2 / var_d[None] + np.log(2*np.pi*var_d[None])) +
        ((data_vr[None] - model_vr[:,None])**2 / var_vr[None] + np.log(2*np.pi*var_vr[None]))
    )
    N = len(ophdata.coord) # number of data points
    K = len(gal) # number of model points

    return logsumexp(chi2, axis=0) - np.log(K)

def distance_ix(c):
    l,b,helio_dist = c.l, c.b, c.distance
    DM = distance_modulus(helio_dist)
    DM_model = 14.58 - (0.2*1/u.deg)*(l - 5*u.deg)
    return (np.abs(DM - DM_model) <= 0.2) & (l > 2*u.deg) & (l < 9*u.deg) & (b > 26.5*u.deg) & (b < 33.5*u.deg)

def main(potential_name, config_filename, results_path=None, overwrite=False):
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

    # path to file to cache the likelihoods
    cache_file = os.path.join(output_path, "ln_likelihoods.npy")

    # load mock streams
    grid = MockStreamGrid.from_config(output_path, config_filename=config_filename,
                                      potential_name=potential_name)
    grid_d = grid.read_cache()
    nstreams = len(grid_d)
    mockstreams = grid_d['w']
    nsuccess = grid_d['success'].sum()

    if nsuccess == 0:
        return

    pot = op.load_potential(potential_name)

    # where to save the plots
    plot_path = os.path.join(output_path, "plots")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # load data
    all_ophdata = OphiuchusData()
    ophdata = OphiuchusData("Name != b'cand26'") # not the star at ~320 km/s
    ndata = len(ophdata.coord)

    # a file to cache the likelihoods
    if not os.path.exists(cache_file) or overwrite:
        logger.debug("Cache file does not exist: {}".format(cache_file))
        lls = np.zeros((nstreams,ndata))
        for i in range(nstreams):
            mockstream = gd.CartesianPhaseSpacePosition.from_w(mockstreams[i].T, units=galactic)
            lls[i] = ln_likelihood(mockstream, ophdata)
        np.save(cache_file, lls)

    else:
        logger.debug("Loading cache file: {}".format(cache_file))
        lls = np.load(cache_file)

    best_ix = lls.sum(axis=1).argmax()
    best_stream = gd.CartesianPhaseSpacePosition.from_w(mockstreams[best_ix].T, units=galactic)

    # first just plot particle positions
    fig = plot_data_stream(all_ophdata, stream=best_stream,
                           stream_style=dict(s=5, color='#aaaaaa', alpha=0.15))
    fig.savefig(os.path.join(plot_path, "best_fit-points.png"), dpi=400)

    # plot particle positions but do observational cut that brani did
    stream_c,_ = best_stream.to_frame(coord.Galactic, galactocentric_frame=galactocentric_frame,
                                      vcirc=vcirc, vlsr=vlsr)
    dist_ix = distance_ix(stream_c)
    fig = plot_data_stream(all_ophdata, stream=best_stream[dist_ix],
                           stream_style=dict(s=5, color='#aaaaaa', alpha=0.15))
    fig.savefig(os.path.join(plot_path, "best_fit-points-cut.png"), dpi=400)

    # plot particle density with observational cut that brani did
    window_ix = ((stream_c.l > 2*u.deg) & (stream_c.l < 9*u.deg) &
                 (stream_c.b > 26.5*u.deg) & (stream_c.b < 33.5*u.deg) &
                 (stream_c.distance > 5.5*u.kpc) & (stream_c.distance < 10*u.kpc))
    grid,log_dens = surface_density(stream_c[window_ix], bandwidth=0.25)

    fig,ax = pl.subplots(1,1,figsize=(6,6))
    cs = ax.contour(grid[:,0].reshape(log_dens.shape), grid[:,1].reshape(log_dens.shape), log_dens,
                    levels=np.arange(-2., 0.5, 0.5), cmap='plasma_r')
    for c in cs.collections:
        c.set_linestyle('solid')

    ax.plot(ophdata.coord.l.degree, ophdata.coord.b.degree,
            marker='o', color='k', linestyle='none')

    ax.set_xlabel("$l$ [deg]", fontsize=18)
    ax.set_ylabel("$b$ [deg]", fontsize=18)
    ax.set_xlim(9,2)
    ax.set_ylim(26.5,33.5)
    ax.set_title("{}: {}".format(potential_name,best_ix), fontsize=22)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_path, "best_fit-density.png"), dpi=400)

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

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(potential_name=args.potential_name, config_filename=args.config_filename,
         results_path=args.results_path, overwrite=args.overwrite)
