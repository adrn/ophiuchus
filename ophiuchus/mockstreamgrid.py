# coding: utf-8

""" Class for running stream generation over the orbit grid """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import numpy as np
import gary.integrate as gi

# Project
from streammorphology.ensemble import align_ensemble
from .ophorbitgridexperiment import OphOrbitGridExperiment
from .mockstream import apw_stream

__all__ = ['MockStreamGrid']

class MockStreamGrid(OphOrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to integrate orbits",
        2: "Unexpected failure."
    }

    _run_kwargs = ['integration_time', 'dt', 'release_every']
    config_defaults = dict(
        integration_time=8192., # Total time to integrate for in Myr
        dt=1., # timestep
        release_every=1, # release a test particle every N timesteps
        w0_filename='w0.npy', # Name of the initial conditions file
        cache_filename='freqmap.npy', # Name of the cache file
        potential_filename='potential.yml' # Name of cached potential file
    )

    def __init__(self, cache_path, overwrite=False, **kwargs):
        super(MockStreamGrid, self).__init__(cache_path, overwrite=overwrite, **kwargs)
        self._nsteps = int(self.config.integration_time / self.config.dt)
        self._nparticles = self._nsteps // self.config.release_every * 2

    @property
    def cache_dtype(self):
        dt = [
            ('dt','f8'),
            ('integration_time','f8'),
            ('release_every','i8'),
            ('w','f8',(self._nparticles+1,6)),
            ('success','b1'),
            ('error_code','i8') # if not successful, why did it fail? see above
        ]
        return dt

    @classmethod
    def run(cls, w0, potential, **kwargs):
        c = dict()
        for k in cls.config_defaults.keys():
            if k not in kwargs:
                c[k] = cls.config_defaults[k]
            else:
                c[k] = kwargs[k]

        dt = c['dt']
        nsteps = int(c['integration_time'] / c['dt'])
        nparticles = nsteps // c['release_every'] * 2

        # return dict
        result = dict()

        # integrate orbit back, then forward again
        torig,w = potential.integrate_orbit(w0.copy(), dt=-dt, nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
        t,w = potential.integrate_orbit(w[-1], dt=dt, t1=torig[-1], nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
        ww = w[:,0].copy()

        prog_mass = np.zeros_like(t) + 1E4
        try:
            stream = apw_stream(potential.c_instance, t, ww,
                                release_every=c['release_every'], G=potential.G,
                                prog_mass=prog_mass)
        except RuntimeError:
            logger.warning("Failed to integrate orbits")
            result['w'] = np.ones((nparticles,6))*np.nan
            result['success'] = False
            result['error_code'] = 1
            return result
        except:
            logger.warning("Unexpected failure!")
            result['w'] = np.ones((nparticles,6))*np.nan
            result['success'] = False
            result['error_code'] = 2
            return result

        allw = align_ensemble(np.vstack((ww[-1], stream))[None])

        result['w'] = allw
        result['dt'] = float(dt)
        result['integration_time'] = c['integration_time']
        result['success'] = True
        result['error_code'] = 0
        return result
