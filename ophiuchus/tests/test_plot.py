# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import matplotlib.pyplot as pl
import astropy.units as u

# Project
from ..data import OphiuchusData
from ..plot import plot_data_orbit

# def test_data():
#     ophdata = OphiuchusData()
#     fig = plot_data_orbit(ophdata)
#     fig.axes[0].set_xlim(-8,1)
#     fig.axes[0].set_ylim(-2,2)
#     pl.show()

def test_data():
    ophdata = OphiuchusData()

    fig,axes = pl.subplots(2,1,sharex=True,figsize=(4,8))
    axes[0].plot(np.cos(ophdata.coord.l), ophdata.coord.distance.value, ls='none')
    axes[1].plot(np.cos(ophdata.coord.l), ophdata.coord.b.value, ls='none')
    # axes[0].set_xlim(axes[0].get_xlim()[::-1])

    fig,axes = pl.subplots(2,1,sharex=True,figsize=(4,8))
    # ax.plot(ophdata.coord_oph.phi1.wrap_at(180*u.deg).degree, ophdata.coord.distance.value, ls='none')
    axes[0].plot(np.cos(ophdata.coord_oph.phi1), ophdata.coord_oph.distance.value, ls='none')
    axes[1].plot(np.cos(ophdata.coord_oph.phi1),
                 ophdata.coord_oph.phi2.degree, ls='none')

    pl.show()
