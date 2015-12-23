# coding: utf-8

""" General utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import gary.integrate as gi
from gary.dynamics.orbit import combine

__all__ = ['integrate_forward_backward']

def integrate_forward_backward(potential, w0, t_forw, t_back, dt=0.5,
                               Integrator=gi.DOPRI853Integrator, t0=0.):
    """
    Integrate an orbit forward and backward from a point and combine
    into a single orbit object.

    Parameters
    ----------
    potential : :class:`gary.potential.PotentialBase`
    w0 : :class:`gary.dynamics.CartesianPhaseSpacePosition`, array_like
    t_forw : numeric
        The amount of time to integate forward in time (a positive number).
    t_back : numeric
        The amount of time to integate backwards in time (a negative number).
    dt : numeric (optional)
        The timestep.
    Integrator : :class:`gary.integrate.Integrator` (optional)
        The integrator class to use.
    t0 : numeric (optional)
        The initial time.

    Returns
    -------
    orbit : :class:`gary.dynamics.CartesianOrbit`
    """

    o1 = potential.integrate_orbit(w0, dt=-dt, t1=t0, t2=t_back, Integrator=Integrator)
    o2 = potential.integrate_orbit(w0, dt=dt, t1=t0, t2=t_forw, Integrator=Integrator)

    o1 = o1[::-1]
    o2 = o2[1:]
    orbit = combine((o1, o2), along_time_axis=True)

    if orbit.pos.shape[-1] == 1:
        return orbit[:,0]
    else:
        return orbit

def get_potential_stream_prog(name):
    """
    Given a potential name, read the mockstream results and return
    the potential object, stream stars, and progenitor orbit.

    """
    pot = op.load_potential(name)

    grid = MockStreamGrid.from_config(cache_path=os.path.join(RESULTSPATH, name, "mockstream"),
                                      config_filename="../../global_mockstream.cfg",
                                      potential_name=name)
    d = grid.read_cache()

    streams = gd.CartesianPhaseSpacePosition.from_w(d['w'].T, units=pot.units)
    dt = d['dt'][0]
    every = d['release_every'][0]
    nsteps = int(np.abs(grid.config.integration_time / dt))

    w0 = np.load(os.path.join(RESULTSPATH, name, "orbitfit", "w0.npy"))[0]
    w0 = gd.CartesianPhaseSpacePosition.from_w(w0, units=pot.units)
    prog = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
    prog = prog[::-1]

    release_t = np.repeat(prog.t[None], axis=0, repeats=2).T.ravel().value

    return pot, streams, prog
