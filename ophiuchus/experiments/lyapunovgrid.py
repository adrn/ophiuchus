# coding: utf-8

""" Class for computing Lyapunov exponents for the orbit grid """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy import log as logger
import numpy as np
import gala.integrate as gi
import gala.dynamics as gd
from gala.units import galactic

# Project
from ..mockstream import ophiuchus_stream
from .core import GridExperiment
from .. import potential as op

__all__ = ['MockStreamGrid']

class LyapunovGrid(GridExperiment):

    # failure error codes
    error_codes = {
        1: "Failed to compute lyapunov exponent",
        2: "Unexpected failure."
    }

    required_kwargs = ['potential_name', 'nperiods', 'nsteps_per_period', 'noffset_orbits']
    config_defaults = {
        'noffset_orbits': 2, # Number of offset orbits to integrate and average.
        'cache_filename': 'lyapunovgrid.npy'
    }

    cache_dtype = [
        ('dt','f8'),
        ('mle_avg','f8'),
        ('mle_end','f8'),
        ('success','b1'),
        ('error_code','i8'), # if not successful, why did it fail? see above
    ]

    @property
    def grid(self):
        if not hasattr(self, '_grid'):
            path = os.path.abspath(os.path.join(self.cache_path, "..", "orbitfit", "w0.npy"))
            self._grid = np.load(path)

        return self._grid

    def run(self, index):
        # return dict
        result = dict()

        nsteps_per_period = self.config.nsteps_per_period
        nperiods = self.config.nperiods
        noffset = self.config.noffset_orbits

        # read potential, initial conditions
        potential = op.load_potential(self.config.potential_name)
        w0 = self.grid[index]

        # I guess this could be determined automatically...but whatever
        T = 200. # The orbits have periods ~200 Myr

        # timestep and number of steps
        dt = T / nsteps_per_period
        nsteps = int(nperiods * nsteps_per_period) # 16384 orbital periods

        try:
            lyap = gd.fast_lyapunov_max(np.ascontiguousarray(w0), potential,
                                        dt=dt, nsteps=nsteps,
                                        noffset_orbits=noffset,
                                        return_orbit=False)
        except RuntimeError:
            logger.warning("Failed to compute lyapunov exponent")
            result['mle_avg'] = np.nan
            result['mle_end'] = np.nan
            result['success'] = False
            result['error_code'] = 1
            return result
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            logger.warning("Unexpected failure: {}".format(str(e)))
            result['mle_avg'] = np.nan
            result['mle_end'] = np.nan
            result['success'] = False
            result['error_code'] = 2
            return result

        # estimate the FTMLE
        lyap = np.mean(lyap, axis=1)
        ix = max(1,nsteps_per_period*nperiods//64)
        FTMLE = np.mean(lyap[-ix:])

        result['dt'] = dt
        result['mle_avg'] = FTMLE.decompose(galactic).value
        result['mle_end'] = lyap[-1].decompose(galactic).value
        result['success'] = True
        result['error_code'] = 0

        return result
