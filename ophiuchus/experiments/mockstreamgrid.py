# coding: utf-8

""" Class for running stream generation over the orbit grid """

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

class MockStreamGrid(OrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to integrate orbits",
        2: "Unexpected failure.",
        3: "Failed to integrate progenitor orbit"
    }

    _run_kwargs = ['integration_time', 'dt', 'release_every', 'w0_path', 'norbits', 'potential_name', 'progenitor_mass']
    config_defaults = dict(
        integration_time=None, # Total time to integrate for in Myr
        dt=1., # timestep
        release_every=None, # release a test particle every N timesteps
        w0_filename='w0.npy', # Name of the initial conditions file
        w0_path='.', # path to initial conditions file, relative to cache path
        norbits=None, # number of orbits to read from the w0 file
        potential_name=None,
        progenitor_mass=None, # mass of the progenitor system
        cache_filename='mockstreamgrid.npy' # Name of the cache file
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
            ('error_code','i8'), # if not successful, why did it fail? see above
            ('progenitor_mass','f8')
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

        # return dict
        result = dict()

        # constant + disruption
        # prog_mass = np.zeros_like(t) + 1E4
        # rr = np.sqrt(np.sum(ww.T[:3]**2,axis=0))
        # peri_ix, = argrelmin(rr)
        # disrupt_idx = peri_ix[-1]
        # if np.abs(peri_ix[-1] - t.size) < 50:
        #     disrupt_idx = peri_ix[-2]

        t_f = result['integration_time'] = -np.abs(c['integration_time'])
        mass = result['progenitor_mass'] = float(c['progenitor_mass'])
        dt = result['dt'] = c['dt']
        every = result['release_every'] = int(c['release_every'])
        try:
            prog,stream = ophiuchus_stream(potential, np.ascontiguousarray(w0.copy()),
                                           t_f=t_f, dt=dt, release_every=every,
                                           prog_mass=mass, Integrator=gi.DOPRI853Integrator,
                                           t_disrupt=-300)
                                           # t_disrupt=t_f) # start disrupted!
        except RuntimeError:
            logger.warning("Failed to integrate orbits")
            # result['w'] = np.ones((nparticles,6))*np.nan
            result['success'] = False
            result['error_code'] = 1
            return result
        except KeyboardInterrupt:
            raise
        except:
            logger.warning("Unexpected failure!")
            # result['w'] = np.ones((nparticles,6))*np.nan
            result['success'] = False
            result['error_code'] = 2
            return result
        allw = np.vstack((prog.w(potential.units)[:,-1].T, stream.w(potential.units).T))

        result['w'] = allw
        result['success'] = True
        result['error_code'] = 0
        return result
