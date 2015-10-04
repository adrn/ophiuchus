# coding: utf-8

""" Class for running frequency mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import cPickle as pickle

# Third-party
import numpy as np
from astropy import log as logger
import gary.integrate as gi
import gary.coordinates as gc
import gary.dynamics as gd
from superfreq import SuperFreq

# Project
from streammorphology.util import estimate_dt_nsteps
from streammorphology.experimentrunner import OrbitGridExperiment
from . import potential as op

__all__ = ['Freqmap']

class OphOrbitGridExperiment(OrbitGridExperiment):

    def _run_wrapper(self, index):
        logger.info("Orbit {0}".format(index))

        # unpack input argument dictionary
        import gary.potential as gp
        potential = gp.load(os.path.join(self.cache_path, self.config.potential_filename),
                            module=op)

        # read out just this initial condition
        norbits = len(self.w0)
        allfreqs = np.memmap(self.cache_file, mode='r',
                             shape=(norbits,), dtype=self.cache_dtype)

        # short-circuit if this orbit is already done
        if allfreqs['success'][index]:
            logger.debug("Orbit {0} already successfully completed.".format(index))
            return None

        # Only pass in things specified in _run_kwargs (w0 and potential required)
        kwargs = dict([(k,self.config[k]) for k in self.config.keys() if k in self._run_kwargs])
        res = self.run(w0=self.w0[index], potential=potential, **kwargs)
        res['index'] = index

        # cache res into a tempfile, return name of tempfile
        tmpfile = os.path.join(self._tmpdir, "{0}-{1}.pickle".format(self.__class__.__name__, index))
        with open(tmpfile, 'w') as f:
            pickle.dump(res, f)
        return tmpfile

class Freqmap(OphOrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to integrate orbit or estimate dt, nsteps.",
        2: "Energy conservation criteria not met.",
        3: "SuperFreq failed on find_fundamental_frequencies().",
        4: "Unexpected failure."
    }

    cache_dtype = [
        ('freqs','f8',(2,3)), # three fundamental frequencies computed in 2 windows
        ('amps','f8',(2,3)), # amplitudes of frequencies in time series
        ('dE_max','f8'), # maximum energy difference (compared to initial) during integration
        ('success','b1'), # whether computing the frequencies succeeded or not
        ('is_tube','b1'), # the orbit is a tube orbit
        ('dt','f8'), # timestep used for integration
        ('nsteps','i8'), # number of steps integrated
        ('error_code','i8') # if not successful, why did it fail? see below
    ]

    _run_kwargs = ['nperiods', 'nsteps_per_period', 'hamming_p', 'energy_tolerance',
                   'force_cartesian', 'nintvec']
    config_defaults = dict(
        energy_tolerance=1E-8, # Maximum allowed fractional energy difference
        nperiods=256, # Total number of orbital periods to integrate for
        nsteps_per_period=512, # Number of steps per integration period for integration stepsize
        hamming_p=4, # Exponent to use for Hamming filter in SuperFreq
        nintvec=15, # maximum number of integer vectors to use in SuperFreq
        force_cartesian=False, # Do frequency analysis on cartesian coordinates
        w0_filename='w0.npy', # Name of the initial conditions file
        cache_filename='freqmap.npy', # Name of the cache file
        potential_filename='potential.yml' # Name of cached potential file
    )

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

        # get timestep and nsteps for integration
        try:
            dt, nsteps = estimate_dt_nsteps(w0.copy(), potential,
                                            c['nperiods'],
                                            c['nsteps_per_period'],
                                            dE_threshold=None)
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            result['freqs'] = np.ones((2,3))*np.nan
            result['success'] = False
            result['error_code'] = 1
            return result
        except:
            logger.warning("Unexpected failure!")
            result['freqs'] = np.ones((2,3))*np.nan
            result['success'] = False
            result['error_code'] = 4
            return result

        # integrate orbit
        logger.debug("Integrating orbit with dt={0}, nsteps={1}".format(dt, nsteps))
        try:
            t,ws = potential.integrate_orbit(w0.copy(), dt=dt, nsteps=nsteps,
                                             Integrator=gi.DOPRI853Integrator,
                                             Integrator_kwargs=dict(atol=1E-11))
        except RuntimeError: # ODE integration failed
            logger.warning("Orbit integration failed.")
            dEmax = 1E10
        else:
            logger.debug('Orbit integrated successfully, checking energy conservation...')

            # check energy conservation for the orbit
            E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
            dE = np.abs(E[1:] - E[0])
            dEmax = dE.max() / np.abs(E[0])
            logger.debug('max(∆E) = {0:.2e}'.format(dEmax))

        # if dEmax > c['energy_tolerance']:
        #     logger.warning("Failed due to energy conservation check.")
        #     result['freqs'] = np.ones((2,3))*np.nan
        #     result['success'] = False
        #     result['error_code'] = 2
        #     result['dE_max'] = dEmax
        #     return result

        # start finding the frequencies -- do first half then second half
        sf1 = SuperFreq(t[:nsteps//2+1], p=c['hamming_p'])
        sf2 = SuperFreq(t[nsteps//2:], p=c['hamming_p'])

        # classify orbit full orbit
        circ = gd.classify_orbit(ws)
        is_tube = np.any(circ)

        # define slices for first and second parts
        sl1 = slice(None,nsteps//2+1)
        sl2 = slice(nsteps//2,None)

        if is_tube and not c['force_cartesian']:
            # first need to flip coordinates so that circulation is around z axis
            new_ws = gd.align_circulation_with_z(ws, circ)
            new_ws = gc.cartesian_to_poincare_polar(new_ws)
            fs1 = [(new_ws[sl1,j] + 1j*new_ws[sl1,j+3]) for j in range(3)]
            fs2 = [(new_ws[sl2,j] + 1j*new_ws[sl2,j+3]) for j in range(3)]

        else:  # box
            fs1 = [(ws[sl1,0,j] + 1j*ws[sl1,0,j+3]) for j in range(3)]
            fs2 = [(ws[sl2,0,j] + 1j*ws[sl2,0,j+3]) for j in range(3)]

        logger.debug("Running SuperFreq on the orbits")
        try:
            freqs1,d1,ixs1 = sf1.find_fundamental_frequencies(fs1, nintvec=c['nintvec'])
            freqs2,d2,ixs2 = sf2.find_fundamental_frequencies(fs2, nintvec=c['nintvec'])
        except:
            result['freqs'] = np.ones((2,3))*np.nan
            result['success'] = False
            result['error_code'] = 3
            return result

        result['freqs'] = np.vstack((freqs1, freqs2))
        result['dE_max'] = dEmax
        result['is_tube'] = float(is_tube)
        result['dt'] = float(dt)
        result['nsteps'] = nsteps
        result['amps'] = np.vstack((d1['|A|'][ixs1], d2['|A|'][ixs2]))
        result['success'] = True
        result['error_code'] = 0
        return result
