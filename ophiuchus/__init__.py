"""
Project package for studying chaos in the Ophiuchus stream
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    import astropy.coordinates as coord
    import astropy.units as u

    # Galactocentric reference frame to use for this project
    galactocentric_frame = coord.Galactocentric(z_sun=0.*u.pc,
                                                galcen_distance=8*u.kpc)
    vcirc = 220.*u.km/u.s
    vlsr = [-11.1, 24, 7.25]*u.km/u.s

    # read from environment variable
    import os
    PROJECTSPATH = os.environ.get('PROJECTSPATH', None)
    if PROJECTSPATH is not None:
        PROJECTSPATH = os.path.abspath(os.path.expanduser(PROJECTSPATH))
        RESULTSPATH = os.path.join(PROJECTSPATH, 'ophiuchus', 'results')
        if not os.path.exists(RESULTSPATH):
            os.mkdir(RESULTSPATH)
    else:
        RESULTSPATH = None

    del coord, u, os

    from . import coordinates
    from . import data
    from . import experiments
    from . import orbitfit
    from . import plot
    from . import potential
    from . import util
