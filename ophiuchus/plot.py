# coding: utf-8

""" Plotting utilities """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as pl
import numpy as np
import gary.coordinates as gc

# Project
from . import galactocentric_frame, vcirc, vlsr
from .coordinates import Ophiuchus

__all__ = ['plot_data_orbit']

default_lims = {
    'phi1': [-10.,10.]*u.deg,
    'phi2': [-2.,2.]*u.deg,
    'l': [9, 2.]*u.deg,
    'b': [27.,33.]*u.deg,
    'distance': [5.5,10]*u.kpc,
    'mul': [-12,0]*u.mas/u.yr,
    'mub': [-4,12]*u.mas/u.yr,
    'vr': [215,335]*u.km/u.s
}
def plot_data_orbit(ophdata, orbit_w=None, stream_coords=False, lims=None, fig=None):
    """
    TODO!
    """

    if lims is None:
        lims = default_lims
    else:
        lims = lims.copy() # so we don't mess with mutable objects

    for k,v in default_lims.items():
        if k not in lims:
            lims[k] = v

    if fig is None:
        fig,_axes = pl.subplots(2,3,figsize=(12,8),sharex=True)
    else:
        _axes = fig.axes
    axes = np.ravel(_axes)
    axes[-1].set_visible(True) # HACK because matplotlib bbox issues with invisible plots

    # plot the data points
    style = dict(marker='o', ls='none', ecolor='#666666', ms=3.)

    if stream_coords:
        x = ophdata.coord_oph.phi1.wrap_at(180*u.deg).to(lims['phi1'].unit).value
        y = ophdata.coord_oph.phi2.to(lims['phi2'].unit).value
        xlim = lims['phi1']
        ylim = lims['phi2']
    else:
        x = ophdata.coord.l.to(lims['l'].unit).value
        y = ophdata.coord.b.to(lims['b'].unit).value
        xlim = lims['l']
        ylim = lims['b']

    # latitude coordinates
    axes[0].errorbar(x, y, 1E-10*x, **style)
    axes[0].set_ylim(*ylim.value)

    axes[1].errorbar(x, ophdata.coord.distance.to(lims['distance'].unit).value,
                     ophdata.coord_err['distance'].to(lims['distance'].unit).value, **style)
    axes[1].set_ylim(*lims['distance'].value)

    for i,k in enumerate(ophdata.veloc.keys()):
        this_lims = lims[k]
        axes[i+2].errorbar(x, ophdata.veloc[k].to(this_lims.unit).value,
                           ophdata.veloc_err[k].to(this_lims.unit).value, **style)
        axes[i+2].set_ylim(*this_lims.value)

    if orbit_w is not None:
        # plot the orbit
        style = dict(marker=None, ls='-')
        w_coord = galactocentric_frame.realize_frame(coord.CartesianRepresentation(orbit_w.T[:3]*u.kpc))\
                                      .transform_to(coord.Galactic)
        w_oph = w_coord.transform_to(Ophiuchus)
        w_vel = gc.vgal_to_hel(w_coord, orbit_w.T[3:]*u.kpc/u.Myr,
                               galactocentric_frame=galactocentric_frame,
                               vcirc=vcirc, vlsr=vlsr)

        x = w_oph.phi1.wrap_at(180*u.deg).to(xlim.unit).value
        axes[0].plot(x, w_oph.phi2.to(ylim.unit).value, **style)
        axes[1].plot(x, w_coord.distance.to(lims['distance'].unit).value, **style)

        for i,k in enumerate(ophdata.veloc.keys()):
            this_lims = lims[k]
            axes[i+2].plot(x, w_vel[i].to(this_lims.unit).value, **style)

    # bottom axis label
    axes[2].set_xlabel(r'$\phi_1$ [deg]')
    axes[3].set_xlabel(r'$\phi_1$ [deg]')
    axes[4].set_xlabel(r'$\phi_1$ [deg]')

    # vertical axis labels
    axes[0].set_ylabel(r'$\phi_2$ [deg]')
    axes[1].set_ylabel(r'$d$ [kpc]')
    axes[2].set_ylabel(r'$\mu_l$ [mas yr$^{-1}$]')
    axes[3].set_ylabel(r'$\mu_b$ [mas yr$^{-1}$]')
    axes[4].set_ylabel(r'$v_{\rm los}$ [km s$^{-1}$]')

    # set all phi1 lims
    axes[0].set_xlim(*xlim.value)
    fig.tight_layout()
    axes[-1].set_visible(False) # HACK because matplotlib bbox issues with invisible plots

    return fig
