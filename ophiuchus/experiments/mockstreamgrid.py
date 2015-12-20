# coding: utf-8

""" Class for running stream generation over the orbit grid """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy import log as logger
import numpy as np
import gary.integrate as gi

# Project
from .core import GridExperiment
from ..mockstream import ophiuchus_stream
from .. import potential as op

__all__ = ['MockStreamGrid']

class MockStreamGrid(GridExperiment):

    # failure error codes
    error_codes = {
        1: "Failed to integrate orbits",
        2: "Unexpected failure.",
        3: "Failed to integrate progenitor orbit"
    }

    required_kwargs = ['integration_time', 'dt', 'release_every', 'potential_name',
                       't_disrupt', 'progenitor_mass']
    config_defaults = {
        "cache_filename": "mockstreamgrid.npy"
    }

    # Grid over initial conditions from the file
    @property
    def grid(self):
        if not hasattr(self, '_grid'):
            path = os.path.abspath(os.path.join(self.cache_path, "..", "orbitfit", "w0.npy"))
            self._grid = np.load(path)
        return self._grid

    def __init__(self, cache_path, overwrite=False, **kwargs):
        super(MockStreamGrid, self).__init__(cache_path, overwrite=overwrite, **kwargs)
        self._nsteps = int(self.config.integration_time / self.config.dt)
        self._nparticles = (self._nsteps+1) // self.config.release_every * 2

    @property
    def cache_dtype(self):
        dt = [
            ('dt','f8'),
            ('integration_time','f8'),
            ('release_every','i8'),
            ('w','f8',(self._nparticles+1,6)),
            ('success','b1'),
            ('error_code','i8')
        ]
        return dt

    def run(self, index):
        # return dict
        result = dict()

        # This experiment grid is over initial conditions
        w0 = self.grid[index]

        # read potential file
        potential = op.load_potential(self.config.potential_name)

        # integration time
        t_disrupt = self.config.t_disrupt * potential.units['time']
        t_f = result['integration_time'] = -np.abs(self.config.integration_time)
        dt = result['dt'] = -np.abs(self.config.dt)
        every = result['release_every'] = int(self.config.release_every)
        mass = self.config.progenitor_mass

        nsteps = int(np.abs(t_f/dt))
        prog = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
        prog = prog[::-1]

        try:
            stream = ophiuchus_stream(potential, prog,
                                      release_every=every,
                                      prog_mass=mass, Integrator=gi.DOPRI853Integrator,
                                      t_disrupt=t_disrupt)
        except RuntimeError:
            logger.warning("Failed to integrate orbits")
            # result['w'] = np.ones((nparticles,6))*np.nan
            result['success'] = False
            result['error_code'] = 1
            return result
        except KeyboardInterrupt:
            raise
        except:
            logger.warning("(internal) Unexpected failure!")
            # result['w'] = np.ones((nparticles,6))*np.nan
            result['success'] = False
            result['error_code'] = 2
            return result
        allw = np.vstack((prog.w(potential.units)[:,-1].T, stream.w(potential.units).T))

        result['w'] = allw
        result['success'] = True
        result['error_code'] = 0

        return result
