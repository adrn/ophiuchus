# coding: utf-8

""" Fit orbit to the Ophiuchus stream members.

Call like:

python fitorbit.py --output-path=../output/orbitfits/ --potential=barred_mw \
-v --nsteps=256 --nwalkers=64 --mpi  --fixtime

"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from six.moves import cPickle as pickle
import os
import sys
import time

# Third-party
from astropy import log as logger
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so
import emcee
from gary.units import galactic
from gary.util import get_pool

# This project
from ophiuchus import RESULTSPATH
import ophiuchus.orbitfit as orbitfit
from ophiuchus.util import integrate_forward_backward
from ophiuchus.data import OphiuchusData
from ophiuchus.plot import plot_data_orbit
import ophiuchus.potential as op

def main(potential_name, dt, mcmc_steps, results_path=None,
         mcmc_walkers=None, mpi=False, overwrite=False, seed=42, continue_mcmc=False,
         fix_integration_time=False, fix_dispersions=False):
    np.random.seed(seed)
    pool = get_pool(mpi=mpi)

    # Load the potential object
    potential = op.load_potential(potential_name)

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

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    sampler_filename = os.path.join(output_path, "sampler.pickle")
    if os.path.exists(sampler_filename) and overwrite:
        logger.debug("Overwriting sampler file: {}".format(sampler_filename))
        time.sleep(0.5)
        os.remove(sampler_filename)

    minimize_filename = os.path.join(output_path, "minimized-params.npy")
    if os.path.exists(minimize_filename) and overwrite:
        logger.debug("Overwriting minimize file: {}".format(minimize_filename))
        os.remove(minimize_filename)

    # ------------------------------------------------------------------------
    # read data
    all_ophdata = OphiuchusData()
    fit_ophdata = OphiuchusData("(source == b'Sesar2015a') | (Name == b'cand9') | (Name == b'cand14')")

    # This is just a good place to initialize from -- I know it sucks to hard-code in
    minimize_p0 = [np.median(fit_ophdata.coord_oph.phi2.decompose(galactic).value)] + \
                  [np.median(fit_ophdata.coord_oph.distance.decompose(galactic).value)] + \
                  [-3.56738886e-02] + \
                  [3.83403870e-03] + \
                  [np.median(fit_ophdata.veloc['vr']).decompose(galactic).value]

    freeze = dict(phi2_sigma=np.radians(0.1),
                  d_sigma=0.025,
                  vr_sigma=0.002)

    if not fix_integration_time:
        minimize_p0 += [3., -3.] # t_forw, t_back
    else:
        freeze['t_forw'] = 3.
        freeze['t_back'] = -3.

    args = (fit_ophdata, potential, dt, freeze)

    if continue_mcmc and not os.path.exists(sampler_filename):
        raise ValueError("Can't continue walkers -- sampler file doesn't exist!")

    if not os.path.exists(sampler_filename) or continue_mcmc:

        if continue_mcmc:
            logger.debug("Loading sampler from: {}".format(sampler_filename))
            with open(sampler_filename, 'rb') as f:
                sampler = pickle.load(f)

            prev_chain = sampler.chain
            prev_mcmc_walkers,prev_mcmc_steps,ndim = prev_chain.shape
            if prev_mcmc_walkers != mcmc_walkers:
                raise ValueError("If continuing walkers, mcmc_walkers ({}) must equal "
                                 "previous mcmc_walkers ({})".format(mcmc_walkers, prev_mcmc_walkers))
            p0 = sampler.chain[:,-1]
            ndim = p0.shape[-1]
            X_minimize = sampler.X_minimize
            args = sampler.args

        else:
            if not os.path.exists(minimize_filename):
                res = so.minimize(lambda *args,**kwargs: -orbitfit.ln_posterior(*args, **kwargs),
                                  x0=minimize_p0, method='powell', args=args)
                X_minimize = res.x
                logger.info("Minimized params: {}".format(X_minimize))
                np.save(minimize_filename, X_minimize)
            X_minimize = np.load(minimize_filename)

            # use output from minimize to initialize MCMC
            _p0 = X_minimize
            ndim = len(_p0)

            if not fix_integration_time:
                ndim += 2

            if not fix_dispersions:
                ndim += 3

            if mcmc_walkers is None:
                mcmc_walkers = ndim*8

            p0 = np.zeros((mcmc_walkers,len(_p0)))
            N = np.random.normal
            p0[:,0] = N(_p0[0], np.radians(0.001), size=mcmc_walkers) # phi2
            p0[:,1] = N(_p0[1], np.median(fit_ophdata.coord_err['distance'].value)/100., size=mcmc_walkers)
            p0[:,2] = N(_p0[2], np.median(fit_ophdata.veloc_err['mul'].value)/1000., size=mcmc_walkers)
            p0[:,3] = N(_p0[3], np.median(fit_ophdata.veloc_err['mub'].value)/1000., size=mcmc_walkers)
            p0[:,4] = N(_p0[4], np.median(fit_ophdata.veloc_err['vr'].value)/100., size=mcmc_walkers)

            if not fix_integration_time:
                p0 = np.hstack((p0, N(_p0[5], 0.01, size=mcmc_walkers)[:,None]))
                p0 = np.hstack((p0, N(_p0[6], 0.01, size=mcmc_walkers)[:,None]))

            if not fix_dispersions:
                for name in ['phi2_sigma', 'd_sigma', 'vr_sigma']:
                    s = freeze.pop(name)
                    p0 = np.hstack((p0, N(s, s/1000., size=mcmc_walkers)[:,None]))

        # get the integration time from minimization or cached on sampler object
        sampler = emcee.EnsembleSampler(nwalkers=mcmc_walkers, dim=ndim,
                                        lnpostfn=orbitfit.ln_posterior,
                                        pool=pool, args=args)

        logger.info("Starting MCMC sampling...")
        _t1 = time.time()
        pos,prob,state = sampler.run_mcmc(p0, mcmc_steps)
        pool.close()
        logger.info("...done sampling after {} seconds.".format(time.time()-_t1))

        # so we can pickle the sampler
        sampler.pool = sampler.lnpostfn = sampler.lnprobfn = None
        sampler.X_minimize = X_minimize

        # collect all chains
        if continue_mcmc:
            sampler._chain = np.hstack((prev_chain, sampler.chain))
            logger.debug("Total mcmc_steps: {}".format(sampler.chain.shape[1]))

        logger.debug("Writing sampler to: {}".format(sampler_filename))
        with open(sampler_filename, 'wb') as f:
            pickle.dump(sampler, f)
    else:
        logger.debug("Loading sampler from: {}".format(sampler_filename))
        with open(sampler_filename, 'rb') as f:
            sampler = pickle.load(f)
        ndim = sampler.dim

    # plot walker trace
    fig,axes = pl.subplots(ndim,1,figsize=(4,3*ndim+1))
    for i in range(ndim):
        axes[i].plot(sampler.chain[...,i].T, drawstyle='steps', color='k', alpha=0.25, marker=None)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "walkers.png"), dpi=300)

    # plot orbit fits
    ix = np.random.randint(len(sampler.flatchain), size=64)
    fig = plot_data_orbit(all_ophdata)
    for sample in sampler.flatchain[ix]:
        sample_w0 = fit_ophdata._mcmc_sample_to_w0(sample[:5])[:,0]
        if fix_integration_time:
            tf,tb = (3.,-3.)
        else:
            tf,tb = (sample[5], sample[6])
        w = integrate_forward_backward(potential, sample_w0, t_forw=tf, t_back=tb)
        fig = plot_data_orbit(all_ophdata, orbit=w, data_style=dict(marker=None),
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

    parser.add_argument("--results-path", dest="results_path", default=None,
                        help="Path to save the output file.")
    parser.add_argument("--potential", dest="potential_name",
                        required=True, help="Name of the potential YAML file.")
    parser.add_argument("--dt", dest="dt", type=float, default=0.5,
                        help="Integration timestep.")

    # emcee
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--mcmc_walkers", dest="mcmc_walkers", type=int, default=None,
                        help="Number of walkers.")
    parser.add_argument("--mcmc_steps", dest="mcmc_steps", type=int, required=True,
                        help="Number of steps to take MCMC.")
    parser.add_argument("--continue", dest="continue_mcmc", default=False,
                        action="store_true", help="Continue sampling from where the sampler left off.")

    parser.add_argument("--fixtime", dest="fixtime", default=False,
                        action="store_true", help="Don't sample over forward/backward integration times.")
    parser.add_argument("--fixdisp", dest="fixdisp", default=False,
                        action="store_true", help="Don't sample over extra dispersions.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.potential_name, dt=args.dt, mcmc_steps=args.mcmc_steps,
         results_path=args.results_path,
         mcmc_walkers=args.mcmc_walkers, mpi=args.mpi, overwrite=args.overwrite,
         continue_mcmc=args.continue_mcmc,
         fix_integration_time=args.fixtime, fix_dispersions=args.fixdisp)

    sys.exit(0)
