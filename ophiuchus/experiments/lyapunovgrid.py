# coding: utf-8

""" Class for computing Lyapunov exponents for the orbit grid """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import numpy as np
import gary.integrate as gi
# from gary.dynamics.mockstream import dissolved_fardal_stream

# Project
from ..mockstream import ophiuchus_stream
from .core import OrbitGridExperiment

__all__ = ['MockStreamGrid']

class LyapunovGrid(OrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to compute lyapunov exponent",
        2: "Unexpected failure."
    }

    _run_kwargs = ['w0_path', 'potential_name']
    config_defaults = dict(
        nperiods=16384, # Total number of orbital periods to integrate for
        nsteps_per_period=512, # Number of steps per integration period for integration stepsize
        noffset_orbits=2, # Number of offset orbits to integrate and average.
        w0_filename='w0.npy', # Name of the initial conditions file
        w0_path='.', # path to initial conditions file, relative to cache path
        potential_name=None,
        cache_filename='lyapunovgrid.npy' # Name of the cache file
    )

    cache_dtype = [
        ('dt','f8'),
        ('mle','f8'),
        ('success','b1'),
        ('error_code','i8'), # if not successful, why did it fail? see above
    ]

    @classmethod
    def run(cls, w0, potential, **kwargs):
        c = dict()
        for k in cls.config_defaults.keys():
            if k not in kwargs:
                c[k] = cls.config_defaults[k]
            else:
                c[k] = kwargs[k]

        # return dict
        result = dict()

        nsteps_per_period = c['nsteps_per_period']
        nperiods = c['nperiods']
        noffset = c['noffset_orbits']

        # I guess this could be determined automatically...but whatever
        T = 200. # The orbits have periods ~200 Myr

        # timestep and number of steps
        dt = T / nsteps_per_period
        nsteps = int(nperiods * nsteps_per_period) # 16384 orbital periods

        try:
            lyap,orbit = gd.fast_lyapunov_max(np.ascontiguousarray(w0), pot,
                                              dt=dt, nsteps=nsteps,
                                              noffset_orbits=noffset)
        except RuntimeError:
            logger.warning("Failed to compute lyapunov exponent")
            result['mle'] = np.nan
            result['success'] = False
            result['error_code'] = 1
            return result
        except KeyboardInterrupt:
            raise
        except:
            logger.warning("Unexpected failure!")
            result['mle'] = np.nan
            result['success'] = False
            result['error_code'] = 2
            return result

        # estimate the FTMLE
        lyap = np.mean(lyap, axis=1)
        FTMLE = np.mean(lyap[-nsteps_per_period*nperiods//10:])

        result['dt'] = dt
        result['mle'] = FTMLE
        result['success'] = True
        result['error_code'] = 0

        return result
