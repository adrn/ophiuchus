# coding: utf-8

""" Fit orbit to the Ophiuchus stream members. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import cPickle as pickle
import os
import sys

# Third-party
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u
import emcee
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so

# Custom
import gary.coordinates as gc
import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp
from gary.units import galactic
import gary.orbitfit as orbitfit
from gary.observation import distance
from gary.util import get_pool

import ophiuchus.potential as op

def main(output_path, potential_file, data_file, sign, dt, nsteps,
         nwalkers=None, mpi=False, overwrite=False, seed=42):
    np.random.seed(seed)
    pool = get_pool(mpi=mpi)

    # Solar position and motion
    reference_frame = dict()
    reference_frame['galactocentric_frame'] = coord.Galactocentric(z_sun=0.*u.pc,
                                                                   galcen_distance=8*u.kpc)
    reference_frame['vcirc'] = 220.*u.km/u.s
    reference_frame['vlsr'] = [-11.1, 24, 7.25]*u.km/u.s

    # Load the potential object
    try:
        potential = gp.load(potential_file)
    except:
        potential = gp.load(potential_file, module=op)

    # filename for saving
    potential_name = os.path.splitext(os.path.basename(potential_file))[0]
    ff = "{}_{}walkers_{}steps_{}sign.pickle".format(potential_name, nwalkers, nsteps, sign)
    output_file = os.path.join(os.path.abspath(output_path), ff)
    if os.path.exists(output_file) and overwrite:
        import time
        time.sleep(0.5)
        os.remove(output_file)

    if os.path.exists(output_file):
        logger.warning("Output file already exists. Exiting.")
        return

    # ------------------------------------------------------------------------
    # read and prepare data
    tbl = np.genfromtxt(data_file, dtype=None, skiprows=1, names=True)

    dists = []
    dist_errs = []
    for DM,err_DM in zip(tbl['DM'], tbl['err_DM']):
        d = distance(np.random.normal(DM, err_DM, size=1024)).to(u.kpc).value
        dists.append(np.median(d))
        dist_errs.append(np.std(d))
    dists = np.array(dists)*u.kpc
    dist_errs = np.array(dist_errs)*u.kpc

    # make astropy coordinate objects and rotate into stream coordinates
    data_coord = coord.ICRS(ra=tbl['ra']*u.degree, dec=tbl['dec']*u.degree, distance=dists)\
                      .transform_to(coord.Galactic)
    R = orbitfit.compute_stream_rotation_matrix(data_coord, align_lon=True)
    data_rot = orbitfit.rotate_sph_coordinate(data_coord, R)

    # containers containing data and uncertainties in correct units
    data = [data_coord.l.decompose(galactic),
            data_coord.b.decompose(galactic),
            dists,
            (tbl['mu_l']*u.mas/u.yr).decompose(galactic),
            (tbl['mu_b']*u.mas/u.yr).decompose(galactic),
            (tbl['v_los']*u.km/u.s).decompose(galactic)]

    errs = [0*data_coord.l.decompose(galactic),
            0*data_coord.b.decompose(galactic),
            dist_errs,
            (tbl['err_mu_l']*u.mas/u.yr).decompose(galactic),
            (tbl['err_mu_b']*u.mas/u.yr).decompose(galactic),
            (tbl['err_v_los']*u.km/u.s).decompose(galactic)]

    # for initial guess for inference, take star with smallest phi1
    ix = data_rot.lon.argmin()
    x0 = (data_rot.lat.decompose(galactic).value[ix],) + \
        tuple([data[j][ix].decompose(galactic).value for j in range(2,6)])

    # initial guess at integration time
    integration_time = 6. # Myr
    args = (data_coord, data[3:], errs, potential, sign*dt, R, reference_frame, np.radians(0.1))
    _p0 = x0 + (sign*integration_time,)

    ndim = len(_p0)
    p0 = np.zeros((nwalkers,ndim))
    p0[:,0] = np.random.normal(_p0[0], np.radians(0.001), size=nwalkers)
    p0[:,1] = np.random.normal(_p0[1], errs[2][ix]/10., size=nwalkers)
    p0[:,2] = np.random.normal(_p0[2], errs[3][ix]/10., size=nwalkers)
    p0[:,3] = np.random.normal(_p0[3], errs[4][ix]/10., size=nwalkers)
    p0[:,4] = np.random.normal(_p0[4], errs[5][ix]/10., size=nwalkers)
    p0[:,5] = np.random.uniform(6., 16., size=nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim,
                                    lnpostfn=orbitfit.ln_posterior,
                                    pool=pool, args=args)

    logger.info("Starting MCMC sampling...")
    pos,prob,state = sampler.run_mcmc(p0, nsteps)
    pool.close()
    logger.info("...done sampling.")

    logger.debug("Writing output to: {}".format(output_file))
    with open(output_file, 'w') as f:
        sampler.lnpostfn = None
        pickle.dump(sampler, f)

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
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")

    parser.add_argument("--output-path", dest="output_path",
                        required=True, help="Path to save the output file.")
    parser.add_argument("--potential-file", dest="potential_file",
                        required=True, help="Name of the potential YAML file.")
    parser.add_argument("--data-file", dest="data_file",
                        required=True, help="Path to the data file for Ophiuchus members.")
    parser.add_argument("--sign", dest="sign", type=int, required=True,
                        help="Integrate forwards or backwards from initial condition.")
    parser.add_argument("--dt", dest="dt", type=float, default=0.5,
                        help="Integration timestep.")

    parser.add_argument("--nwalkers", dest="nwalkers", type=int, default=None,
                        help="Number of walkers.")
    parser.add_argument("--nsteps", dest="nsteps", type=int, required=True,
                        help="Number of steps to take MCMC.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    try:
        main(args.output_path, args.potential_file, data_file=args.data_file,
             sign=args.sign, dt=args.dt, nsteps=args.nsteps,
             nwalkers=args.nwalkers, mpi=args.mpi, overwrite=args.overwrite)
    except:
        sys.exit(1)

    sys.exit(0)
