# Licensed under a 3-clause BSD style license - see LICENSE.rst

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
    from .potential import OphiuchusPotential

    # These are the canonical potentials I'll use in the paper:
    #   - One Milky Way model with a time-dependent Bar
    #   - One axisymmetric, static Milky Way model
    barred_mw = OphiuchusPotential()
    static_mw = OphiuchusPotential(bar=dict(m=0.),
                                   spheroid=dict(m=1E10, c=0.2),
                                   disk=dict(m=6.E10))
