# coding: utf-8

""" Fit orbit to the Ophiuchus stream members. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import cPickle as pickle
import os
import sys
import time

# Third-party
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so
import emcee

# Custom
import gary.integrate as gi
import gary.coordinates as gc
import gary.potential as gp
from gary.units import galactic
from gary.dynamics import orbitfit
from gary.observation import distance
from gary.util import get_pool

from ophiuchus import galactocentric_frame, vcirc, vlsr
import ophiuchus.potential as op

lon_lims = (-0.2, 6.2) # deg
dist_lims = (5.5, 10.) # kpc
vlos_lims = (230, 325.) # km/s
def plot_data_orbit(data, errs, data_coord, data_rot, w0_obs, potential, R, integration_time=-15.):
    # the fit initial conditions in rotated stream coordinates
    phi2_0 = w0_obs[0]*u.radian
    dist_0 = w0_obs[1]*u.kpc
    mul_0 = w0_obs[2]*u.rad/u.Myr
    mub_0 = w0_obs[3]*u.rad/u.Myr
    vr_0 = w0_obs[4]*u.kpc/u.Myr

    # convert position from stream coordinates to data coordinate frame
    sph = coord.SphericalRepresentation(lon=0.*u.radian, lat=phi2_0, distance=dist_0)
    xyz = sph.represent_as(coord.CartesianRepresentation).xyz.value
    in_frame_car = coord.CartesianRepresentation(R.T.dot(xyz).T*u.kpc)
    initial_coord = data_coord.realize_frame(in_frame_car)

    # now convert to galactocentric coordinates
    x0 = initial_coord.transform_to(galactocentric_frame).cartesian.xyz.decompose(galactic).value
    v0 = gc.vhel_to_gal(initial_coord, pm=(mul_0,mub_0), rv=vr_0,
                        galactocentric_frame=galactocentric_frame,
                        vcirc=vcirc, vlsr=vlsr).decompose(galactic).value
    w0 = np.append(x0, v0)

    t,w = potential.integrate_orbit(w0, dt=np.sign(integration_time)*0.1, t1=0., t2=integration_time, Integrator=gi.DOPRI853Integrator)
    w = w[:,0]
    w_coord = galactocentric_frame.realize_frame(coord.CartesianRepresentation(w[:,:3].T*u.kpc)).transform_to(data_coord)
    w_rot_coord = orbitfit.rotate_sph_coordinate(w_coord, R)
    w_vel = gc.vgal_to_hel(w_coord, w[:,3:].T*u.kpc/u.Myr,
                           galactocentric_frame=galactocentric_frame, vcirc=vcirc, vlsr=vlsr)

    fig,axes = pl.subplots(5,1,figsize=(4,15),sharex=True)

    # data
    x = data_rot.lon.degree
#     x = np.cos(data_rot.lon)
    axes[0].plot(x, data_rot.lat.degree, marker='o', ls='none')
#     axes[0].plot(x, np.cos(data_rot.lat), marker='o', ls='none')
    axes[1].errorbar(x, data[2].value, errs[2].value, marker='o', ls='none')
    axes[2].errorbar(x, data[3].to(u.mas/u.yr).value, errs[3].to(u.mas/u.yr).value, marker='o', ls='none')
    axes[3].errorbar(x, data[4].to(u.mas/u.yr).value, errs[4].to(u.mas/u.yr).value, marker='o', ls='none')
    axes[4].errorbar(x, data[5].to(u.km/u.s).value, errs[5].to(u.km/u.s).value, marker='o', ls='none')

    # orbit
    x = w_rot_coord.lon.degree
#     x = np.cos(w_rot_coord.lon)
    axes[0].plot(x, w_rot_coord.lat.degree, marker=None)
#     axes[0].plot(x, np.cos(w_rot_coord.lat), marker=None)
    axes[1].plot(x, w_rot_coord.distance.decompose(galactic).value, marker=None)
    axes[2].plot(x, w_vel[0].to(u.mas/u.yr).value, marker=None)
    axes[3].plot(x, w_vel[1].to(u.mas/u.yr).value, marker=None)
    axes[4].plot(x, w_vel[2].to(u.km/u.s).value, marker=None)

#     axes[-1].set_xlabel(r'$\cos(\phi_1)$')
    axes[-1].set_xlabel(r'$\phi_1$ [deg]')
    axes[0].set_ylabel(r'$\phi_2$ [deg]')
    axes[1].set_ylabel(r'$d$ [kpc]')
    axes[2].set_ylabel(r'$\mu_l$ [mas yr$^{-1}$]')
    axes[3].set_ylabel(r'$\mu_b$ [mas yr$^{-1}$]')
    axes[4].set_ylabel(r'$v_{\rm los}$ [km s$^{-1}$]')

    axes[0].set_xlim(lon_lims)
    axes[0].set_ylim(-1,1)

    axes[1].set_ylim(dist_lims)
    axes[2].set_ylim(-12, 0)
    axes[3].set_ylim(-2, 8)
    axes[4].set_ylim(vlos_lims)

    return fig, w0

class LnPostWrapper(object):
    def __init__(self, integration_time):
        self.integration_time = integration_time
    def __call__(self, p, *args, **kwargs):
        return orbitfit.ln_posterior(list(p)+[self.integration_time],*args,**kwargs)

def main(top_output_path, potential_file, data_file, sign, dt,
         nsteps, nwalkers=None, mpi=False, overwrite=False, seed=42, continue_mcmc=False):
    np.random.seed(seed)
    pool = get_pool(mpi=mpi)

    # Solar position and motion
    reference_frame = dict()
    reference_frame['galactocentric_frame'] = galactocentric_frame
    reference_frame['vcirc'] = vcirc
    reference_frame['vlsr'] = vlsr

    # Load the potential object
    try:
        potential = gp.load(potential_file)
    except:
        potential = gp.load(potential_file, module=op)

    # top-level output path for saving (this will create a subdir within output_path)
    top_output_path = os.path.abspath(os.path.expanduser(top_output_path))
    potential_name = os.path.splitext(os.path.basename(potential_file))[0]
    output_path = os.path.join(top_output_path, potential_name)
    logger.debug("Output path: {}".format(output_path))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    sampler_filename = os.path.join(output_path, "sampler.pickle")
    if os.path.exists(sampler_filename) and overwrite:
        time.sleep(0.5)
        os.remove(sampler_filename)

    # ------------------------------------------------------------------------
    # read and prepare data
    tbl = np.genfromtxt(data_file, dtype=None, skip_header=2, names=True)

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
    R = orbitfit.compute_stream_rotation_matrix(data_coord, align_lon='max')
    data_rot = orbitfit.rotate_sph_coordinate(data_coord, R)

    # containers for data and uncertainties in correct units
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

    # if not os.path.exists(minimize_file):
    # for initial guess for inference, take the star with smallest phi1 as the pivot
    ix = data_rot.lon.argmin() # HACK: should be customizable
    x0 = (data_rot.lat.decompose(galactic).value[ix],) + \
        tuple([data[j][ix].decompose(galactic).value for j in range(2,6)])

    # initial guess at integration time
    integration_time = 6. # Myr

    # first minimize
    p0 = tuple(x0) + (sign*integration_time,)
    args = (data_coord,
            [d for d in data[3:]],
            [e for e in errs],
            potential, sign*dt, R, reference_frame,
            np.radians(0.1), 0.025, 0.002) # phi2_sigma, d_sigma, vlos_sigma

    if continue_mcmc and not os.path.exists(sampler_filename):
        raise ValueError("Can't continue walkers -- sampler file doesn't exist!")

    if not os.path.exists(sampler_filename) or continue_mcmc:

        if continue_mcmc:
            logger.debug("Loading sampler from: {}".format(sampler_filename))
            with open(sampler_filename, 'r') as f:
                sampler = pickle.load(f)

            prev_chain = sampler.chain
            prev_nwalkers,prev_nsteps,ndim = prev_chain.shape
            if prev_nwalkers != nwalkers:
                raise ValueError("If continuing walkers, nwalkers ({}) must equal "
                                 "previous nwalkers ({})".format(nwalkers, prev_nwalkers))
            p0 = sampler.chain[:,-1]
            ndim = p0.shape[-1]
            X_minimize = sampler.X_minimize

        else:
            # res = so.minimize(lambda *args,**kwargs: -orbitfit.ln_posterior(*args, **kwargs),
            #                   x0=p0, method='powell', args=args)
            # X_minimize = res.x
            # logger.info("Minimized params: {}".format(X_minimize))

            # TESTING
            X_minimize = [-2.30396525e-03, 8.75970429e+00, -2.72026927e-02, 1.01370688e-02, 2.92748461e-01, -15.]

            # use output from minimize to initialize MCMC
            _p0 = X_minimize[:-1]
            ndim = len(_p0)

            if nwalkers is None:
                nwalkers = ndim*8
            p0 = np.zeros((nwalkers,ndim))
            p0[:,0] = np.random.normal(_p0[0], np.radians(0.001), size=nwalkers)
            p0[:,1] = np.random.normal(_p0[1], errs[2][ix]/100., size=nwalkers)
            p0[:,2] = np.random.normal(_p0[2], errs[3][ix]/10000., size=nwalkers)
            p0[:,3] = np.random.normal(_p0[3], errs[4][ix]/10000., size=nwalkers)
            p0[:,4] = np.random.normal(_p0[4], errs[5][ix]/100., size=nwalkers)

        # get the integration time from minimization or cached on sampler object
        integ_time = X_minimize[5]
        ln_posterior = LnPostWrapper(integ_time)
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim,
                                        lnpostfn=ln_posterior,
                                        pool=pool, args=args)

        logger.info("Starting MCMC sampling...")
        _t1 = time.time()
        pos,prob,state = sampler.run_mcmc(p0, nsteps)
        pool.close()
        logger.info("...done sampling after {} seconds.".format(time.time()-_t1))

        # so we can pickle the sampler
        sampler.pool = sampler.lnpostfn = sampler.lnprobfn = None
        sampler.X_minimize = X_minimize

        # collect all chains
        if continue_mcmc:
            sampler._chain = np.hstack((prev_chain, sampler.chain))
            logger.debug("Total nsteps: {}".format(sampler.chain.shape[1]))

        logger.debug("Writing sampler to: {}".format(sampler_filename))
        with open(sampler_filename, 'w') as f:
            pickle.dump(sampler, f)
    else:
        logger.debug("Loading sampler from: {}".format(sampler_filename))
        with open(sampler_filename, 'r') as f:
            sampler = pickle.load(f)

    # plot walker trace
    fig,axes = pl.subplots(ndim,1,figsize=(4,3*ndim+1))
    for i in range(ndim):
        axes[i].plot(sampler.chain[...,i].T, drawstyle='steps', color='k', alpha=0.25, marker=None)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "walkers.png"), dpi=300)

    X2 = np.median(sampler.flatchain, axis=0)
    print(X2)
    fig,w0 = plot_data_orbit(data, errs, data_coord, data_rot, X2, potential, R, integration_time=integ_time)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "median-orbit.png"), dpi=300)

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
    parser.add_argument("--potential-file", dest="potential_file",
                        required=True, help="Name of the potential YAML file.")
    parser.add_argument("--data-file", dest="data_file",
                        required=True, help="Path to the data file for Ophiuchus members.")
    parser.add_argument("--sign", dest="sign", type=int, required=True,
                        help="Integrate forwards or backwards from initial condition.")
    parser.add_argument("--dt", dest="dt", type=float, default=0.5,
                        help="Integration timestep.")

    # emcee
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--nwalkers", dest="nwalkers", type=int, default=None,
                        help="Number of walkers.")
    parser.add_argument("--nsteps", dest="nsteps", type=int, required=True,
                        help="Number of steps to take MCMC.")
    parser.add_argument("--continue", dest="continue_mcmc", default=False,
                        action="store_true", help="Continue sampling from where the sampler left off.")

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
             nwalkers=args.nwalkers, mpi=args.mpi, overwrite=args.overwrite,
             continue_mcmc=args.continue_mcmc)
    except:
        logger.error("Unexpected error! {}: {}".format(*sys.exc_info()))
        sys.exit(1)

    sys.exit(0)
