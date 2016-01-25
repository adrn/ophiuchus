.. _coordinates:

******************************************************
Ophiuchus stream coordinates (`ophiuchus.coordinates`)
******************************************************

Introduction
============

.. _coordinates-api:

This sub-package contains a :class:`~astropy.coordinates.BaseCoordinateFrame` object to be used with the `astropy.coordinates <http://docs.astropy.org/en/latest/coordinates/index.html>`_ framework for transforming coordinates into this Heliocentric coordinate system aligned with the Ophiuchus stream.

Getting started
---------------

First some imports:

    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> import matplotlib.pyplot as pl
    >>> import numpy as np
    >>> from ophiuchus.coordinates import Ophiuchus

To transform a set or catalog of coordinates to the Ophiuchus system, you must first define an Astropy coordinate object. We'll generate some random data around the position of the stream for this example:

    >>> l = np.random.normal(5.5, 1., size=128)*u.degree
    >>> b = np.random.normal(33., 0.1, size=l.size)*u.degree
    >>> c = coord.SkyCoord(l=l, b=b, frame='galactic')
    >>> c_oph = c.transform_to(Ophiuchus)

Let's plot the coordinates in the new system:

    >>> fig,ax = pl.subplots(1,1,figsize=(10,4),subplot_kw=dict(projection='hammer'))
    >>> ax.plot(c_oph.phi1.radian, c_oph.phi2.radian, ls='none', marker='o', alpha=0.25)
    >>> ax.set_xlabel(r'$\phi_1$')
    >>> ax.set_ylabel(r'$\phi_2$')
    >>> ax.grid()

.. plot::
    :align: center

    import astropy.coordinates as coord
    import astropy.units as u
    import matplotlib.pyplot as pl
    import numpy as np
    from ophiuchus.coordinates import Ophiuchus

    l = np.random.normal(5.5, 25., size=128)*u.degree
    b = np.random.normal(33., 0.1, size=l.size)*u.degree
    c = coord.SkyCoord(l=l, b=b, frame='galactic')
    c_oph = c.transform_to(Ophiuchus)

    fig,ax = pl.subplots(1,1,figsize=(10,4),subplot_kw=dict(projection='hammer'))
    ax.plot(c_oph.phi1.radian, c_oph.phi2.radian, ls='none', marker='o', alpha=0.25)
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')
    ax.grid()

API
===

.. automodapi:: ophiuchus.coordinates
    :no-inheritance-diagram:

