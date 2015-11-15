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
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so
import emcee
import gary.potential as gp
from gary.units import galactic
from gary.util import get_pool

# This project
from ophiuchus import orbitfit
from ophiuchus.util import integrate_forward_backward
from ophiuchus.data import OphiuchusData
from ophiuchus.plot import plot_data_orbit
import ophiuchus.potential as op

def main(top_output_path, potential_file, dt,
         nsteps, nwalkers=None, mpi=False, overwrite=False, seed=42, continue_mcmc=False):
    np.random.seed(seed)
    pool = get_pool(mpi=mpi)

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
        logger.debug("Overwriting sampler file: {}".format(sampler_filename))
        time.sleep(0.5)
        os.remove(sampler_filename)

    minimize_filename = os.path.join(output_path, "minimized.npy")
    if os.path.exists(minimize_filename) and overwrite:
        logger.debug("Overwriting minimize file: {}".format(minimize_filename))
        os.remove(minimize_filename)

    # ------------------------------------------------------------------------
    # read data
    all_ophdata = OphiuchusData()
    fit_ophdata = OphiuchusData("(source == 'Sesar2015a') | (Name == 'cand9') | (Name == 'cand14')")

    # This is just a good place to initialize from -- I know it sucks to hard-code in
    p0 = [np.median(fit_ophdata.coord_oph.phi2.decompose(galactic).value)] + \
         [np.median(fit_ophdata.coord_oph.distance.decompose(galactic).value)] + \
         [-3.56738886e-02] + \
         [3.83403870e-03] + \
         [np.median(fit_ophdata.veloc['vr']).decompose(galactic).value] + \
         [3., -3.] # t_forw, t_back

    freeze = dict(phi2_sigma=np.radians(0.1),
                  d_sigma=0.025,
                  vr_sigma=0.002)
    args = (fit_ophdata, potential, dt, freeze)

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
            args = sampler.args

        else:
            if not os.path.exists(minimize_filename):
                res = so.minimize(lambda *args,**kwargs: -orbitfit.ln_posterior(*args, **kwargs),
                                  x0=p0, method='powell', args=args)
                X_minimize = res.x
                logger.info("Minimized params: {}".format(X_minimize))
                np.save(minimize_filename, X_minimize)
            X_minimize = np.load(minimize_filename)

            # use output from minimize to initialize MCMC
            _p0 = X_minimize[:-1]
            ndim = len(_p0)

            if nwalkers is None:
                nwalkers = ndim*8
            p0 = np.zeros((nwalkers,ndim))
            N = np.random.normal
            p0[:,0] = N(_p0[0], np.radians(0.001), size=nwalkers) # phi2
            p0[:,1] = N(_p0[1], np.median(fit_ophdata.coord_err['distance'].value)/100., size=nwalkers)
            p0[:,2] = N(_p0[2], np.median(fit_ophdata.veloc_err['mul'].value)/1000., size=nwalkers)
            p0[:,3] = N(_p0[3], np.median(fit_ophdata.veloc_err['mub'].value)/1000., size=nwalkers)
            p0[:,4] = N(_p0[4], np.median(fit_ophdata.veloc_err['vr'].value)/100., size=nwalkers)

        # get the integration time from minimization or cached on sampler object
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim,
                                        lnpostfn=orbitfit.ln_posterior,
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

    # plot orbit fits
    ix = np.random.randint(len(sampler.flatchain), size=128)
    fig = plot_data_orbit(all_ophdata)
    for sample in sampler.flatchain[ix]:
        sample_w0 = sample[:5]
        w = integrate_forward_backward(potential, sample_w0, t_forw=sample[5], t_back=sample[6])
        fig = plot_data_orbit(all_ophdata, orbit_w=w, data_style=dict(marker=None),
                              orbit_style=dict(color='#2166AC', alpha=0.1))

    fig.tight_layout()
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

    main(args.output_path, args.potential_file, data_file=args.data_file,
         sign=args.sign, dt=args.dt, nsteps=args.nsteps,
         nwalkers=args.nwalkers, mpi=args.mpi, overwrite=args.overwrite,
         continue_mcmc=args.continue_mcmc)

    sys.exit(0)
