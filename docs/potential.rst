.. _potential:

**************************************************
Milky Way potential models (`ophiuchus.potential`)
**************************************************

Introduction
============

.. _potential-api:

The gravitational potentials used in the paper are implemented as :mod:`gary.potential` and the potentials instantiated with the specific parameter choices used in the paper can be loaded by name as:

    >>> import ophiuchus.potential as op
    >>> pot = op.load_potential('static_mw')
    >>> pot = op.load_potential('barred_mw_1')

For the barred potentials 1-9, replace 1 above with the desired number.

You can also create your own potential with a specific parameter choice using the :class:`ophiuchus.potential.OphiuchusPotential` class. This class accepts four dictionaries defining parameter choices for the disk, halo, bar, and spheroid component of the potential:

    >>> import astropy.units as u
    >>> from gary.units import galactic
    >>> disk = dict(m=6E10*u.Msun, a=3.*u.kpc, b=0.28*u.kpc)
    >>> halo = dict(v_c=200*u.km/u.s, r_s=30*u.kpc, q_z=1.)
    >>> bar = dict(m=2E10*u.Msun, r_s=1.0,
    ...            Omega=40*u.km/u.s/u.kpc, alpha=20*u.degree)
    >>> spher = dict(m=0., c=1.)
    >>> pot = op.OphiuchusPotential(units=galactic,
    ...                             disk=disk, halo=halo,
    ...                             bar=bar, spheroid=spher)

This potential object acts like any of the :mod:`gary.potential` objects -- for example, we can easily integrate orbits:

    >>> import gary.integrate as gi
    >>> w0 = gd.CartesianPhaseSpacePosition(pos=([[5.,0,0],
    ...                                           [2.,0,0]]*u.kpc).T,
    ...                                     vel=([[0.,200,25],
    ...                                           [0.,150,150]]*u.km/u.s).T)
    >>> orbits = pot.integrate_orbit(w0, dt=0.5, nsteps=10000,
    ...                              Integrator=gi.DOPRI853Integrator)
    >>> fig = orbits.plot(linestyle='none', marker='.', alpha=0.1)

.. plot::
    :align: center

    import astropy.units as u
    import matplotlib.pyplot as pl
    import numpy as np
    import gary.dynamics as gd
    import gary.integrate as gi
    from gary.units import galactic
    import ophiuchus.potential as op

    disk = dict(m=6E10*u.Msun, a=3.*u.kpc, b=0.28*u.kpc)
    halo = dict(v_c=200*u.km/u.s, r_s=30*u.kpc, q_z=1.)
    bar = dict(m=2E10*u.Msun, r_s=1.0, Omega=40*u.km/u.s/u.kpc, alpha=20*u.degree)
    spher = dict(m=0., c=1.)
    pot = op.OphiuchusPotential(units=galactic, disk=disk, halo=halo, bar=bar, spheroid=spher)

    w0 = gd.CartesianPhaseSpacePosition(pos=([[5.,0,0],[2.,0,0]]*u.kpc).T,
                                        vel=([[0.,200,25],[0.,150,150]]*u.km/u.s).T)
    orbits = pot.integrate_orbit(w0, dt=0.5, nsteps=10000, Integrator=gi.DOPRI853Integrator)

    fig = orbits.plot(linestyle='none', marker='.', alpha=0.1)

API
===

.. automodapi:: ophiuchus.potential
    :no-inheritance-diagram:

