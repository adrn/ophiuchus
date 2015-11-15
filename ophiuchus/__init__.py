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

    del coord, u

    from .util import *
