.. _read-data-integrate-plot:

**************************************************
Example: read BHB star data, integrate some orbits
**************************************************

In this example, we'll use the package to read in the Blue Horizontal Branch (BHB) star data from `Sesar et al. (2015) <http://arxiv.org/abs/1501.00581>`_ and `Sesar et al. (2016) <http://arxiv.org/abs/1512.00469>`_, integrate some orbits, and transform to Ophiuchus coordinates to make some plots.

First, some imports:

    >>> import astropy.units as u
    >>> import gary.integrate as gi
    >>> import gary.dynamics as gd
    >>> import gary.coordinates as gc
    >>> from gary.units import galactic
    >>> import ophiuchus.potential as op
    >>> from ophiuchus.coordinates import Ophiuchus

The package also contains a class that contains the BHB star data:

    >>> from ophiuchus.data import OphiuchusData

We can use this class to read all data, or only the data from a specific publication:

    >>> data = OphiuchusData() # all data
    >>> S15_data = OphiuchusData(expr='source == b"Sesar2015a"')

To transform from Heliocentric coordinates to Galactocentric coordinates, we must define a reference frame. The location and velocity of the Sun that was used in this paper are defined in the following variables:

    >>> from ophiuchus import galactocentric_frame, vcirc, vlsr

which can be passed in to the Astropy and Gary coordinate transformation functions to transform the positions and velocities of the BHB stars to a Galactocentric frame:

    >>> pos = data.coord.transform_to(galactocentric_frame).cartesian.xyz
    >>> vel = gc.vhel_to_gal(data.coord, rv=data.veloc['vr'],
    ...                      pm=(data.veloc['mul'],data.veloc['mub']),
    ...                      galactocentric_frame=galactocentric_frame,
    ...                      vcirc=vcirc, vlsr=vlsr)
    >>> pl.plot(pos[0], pos[2], linestyle='none', marker='o')
    >>> pl.xlabel("x [kpc]")
    >>> pl.ylabel("z [kpc]")

.. plot::
    :align: center

    import matplotlib.pyplot as pl
    import gary.coordinates as gc
    from ophiuchus.data import OphiuchusData
    from ophiuchus import galactocentric_frame, vcirc, vlsr

    data = OphiuchusData()

    pos = data.coord.transform_to(galactocentric_frame).cartesian.xyz
    vel = gc.vhel_to_gal(data.coord, rv=data.veloc['vr'],
                         pm=(data.veloc['mul'],data.veloc['mub']),
                         galactocentric_frame=galactocentric_frame,
                         vcirc=vcirc, vlsr=vlsr)
    pl.plot(pos[0], pos[2], linestyle='none', marker='o')
    pl.xlabel("x [kpc]")
    pl.ylabel("z [kpc]")
    pl.tight_layout()

We could now take these positions and velocities and integrate some orbits in one of the potentials. Of course, the uncertainties on the kinematics are large so just integrating the mean orbits is not very useful. But, we'll do it for visualization anyways. First we'll read the potential objects:

    >>> barred_potential = op.load_potential('barred_mw_8')
    >>> static_potential = op.load_potential('static_mw')

Now we'll create a :class:`~gary.dynamics.CartesianPhaseSpacePosition` object from the positions and velocities of all of the stars and integrate these orbits backwards in time (for 1 Gyr) in a barred potential and a static potential model for the Milky Way. We'll use the more precise :class:`~gary.integrate.DOPRI853Integrator`:

    >>> data_w = gd.CartesianPhaseSpacePosition(pos=pos, vel=vel)
    >>> barred_orbits = barred_potential.integrate_orbit(data_w, dt=-0.5, nsteps=2000,
    ...                                                  Integrator=gi.DOPRI853Integrator)
    >>> static_orbits = static_potential.integrate_orbit(data_w, dt=-0.5, nsteps=2000,
    ...                                                  Integrator=gi.DOPRI853Integrator)
    >>> barred_fig = barred_orbits.plot()
    >>> barred_fig.axes[1].set_title("Barred potential", fontsize=20)
    >>> barred_fig.axes[1].set_xlim(-20,20); barred_fig.axes[1].set_ylim(-20,20)
    >>> static_fig = static_orbits.plot()
    >>> static_fig.axes[1].set_title("Static potential", fontsize=20)
    >>> static_fig.axes[1].set_xlim(-20,20); static_fig.axes[1].set_ylim(-20,20)

.. plot::
    :align: center

    import gary.integrate as gi
    import gary.coordinates as gc
    import gary.dynamics as gd
    import matplotlib.pyplot as pl
    from ophiuchus.data import OphiuchusData
    from ophiuchus import galactocentric_frame, vcirc, vlsr
    import ophiuchus.potential as op

    data = OphiuchusData()

    pos = data.coord.transform_to(galactocentric_frame).cartesian.xyz
    vel = gc.vhel_to_gal(data.coord, rv=data.veloc['vr'],
                         pm=(data.veloc['mul'],data.veloc['mub']),
                         galactocentric_frame=galactocentric_frame,
                         vcirc=vcirc, vlsr=vlsr)

    barred_potential = op.load_potential('barred_mw_8')
    static_potential = op.load_potential('static_mw')

    data_w = gd.CartesianPhaseSpacePosition(pos=pos, vel=vel)
    barred_orbits = barred_potential.integrate_orbit(data_w, dt=-0.5, nsteps=2000,
                                                     Integrator=gi.DOPRI853Integrator)
    static_orbits = static_potential.integrate_orbit(data_w, dt=-0.5, nsteps=2000,
                                                     Integrator=gi.DOPRI853Integrator)
    barred_fig = barred_orbits.plot()
    barred_fig.axes[1].set_title("Barred potential", fontsize=20)
    barred_fig.axes[1].set_xlim(-20,20)
    barred_fig.axes[1].set_ylim(-20,20)

    static_fig = static_orbits.plot()
    static_fig.axes[1].set_title("Static potential", fontsize=20)
    static_fig.axes[1].set_xlim(-20,20)
    static_fig.axes[1].set_ylim(-20,20)
