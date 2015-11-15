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
default_data_style = {
    'marker': 'o',
    'linestyle': 'none',
    'ecolor': '#666666',
    'markersize': 3.
}
default_orbit_style = {
    'marker': None,
    'linestyle': '-',
}
def plot_data_orbit(ophdata, orbit_w=None, use_stream_coords=False, lims=None,
                    fig=None, data_style=None, orbit_style=None):
    """
    Plot the Ophiuchus stream data and optionally overplot an orbit in either
    galactic or stream coordinates.

    Parameters
    ----------
    ophdata : :class:`ophiuchus.data.OphiuchusData`
    orbit_w : :class:`numpy.ndarray` (optional)
    use_stream_coords : bool (optional)
        Plot things in terms of rotated stream coordinates. Default is False,
        will plot in terms of Galactic coordinates.
    lims : dict (optional)
        A dictionary of axis limits -- must contain units. The units specified
        in the axis limits will be the displayed units of the data.
    fig : :class:`matplotlib.Figure` (optional)
        Overplot multiple datasets on one figure.
    data_style : dict (optional)
        Dictionary of keyword style arguments passed to
        :func:`matplotlib.errorbar` for the data points.
    orbit_style : dict (optional)
        Dictionary of keyword style arguments passed to
        :func:`matplotlib.plot` for the orbit line.

    Returns
    -------
    fig : :class:`matplotlib.Figure`

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
    if data_style is None:
        data_style = default_data_style
    else:
        data_style = data_style.copy() # so we don't mess with mutable objects
    for k,v in default_data_style.items():
        if k not in data_style:
            data_style[k] = v

    if use_stream_coords:
        x = ophdata.coord_oph.phi1.wrap_at(180*u.deg).to(lims['phi1'].unit).value
        y = ophdata.coord_oph.phi2.to(lims['phi2'].unit).value
        xlim = lims['phi1']
        ylim = lims['phi2']
        xlabel = r'$\phi_1$ [deg]'
        yylabel = r'$\phi_2$ [deg]'
    else:
        x = ophdata.coord.l.to(lims['l'].unit).value
        y = ophdata.coord.b.to(lims['b'].unit).value
        xlim = lims['l']
        ylim = lims['b']
        xlabel = r'$l$ [deg]'
        yylabel = r'$b$ [deg]'

    # latitude coordinates
    axes[0].errorbar(x, y, 1E-10*x, **data_style)
    axes[0].set_ylim(*ylim.value)

    axes[1].errorbar(x, ophdata.coord.distance.to(lims['distance'].unit).value,
                     ophdata.coord_err['distance'].to(lims['distance'].unit).value, **data_style)
    axes[1].set_ylim(*lims['distance'].value)

    for i,k in enumerate(ophdata.veloc.keys()):
        this_lims = lims[k]
        axes[i+2].errorbar(x, ophdata.veloc[k].to(this_lims.unit).value,
                           ophdata.veloc_err[k].to(this_lims.unit).value, **data_style)
        axes[i+2].set_ylim(*this_lims.value)

    if orbit_w is not None:
        # plot the orbit
        if orbit_style is None:
            orbit_style = default_orbit_style
        else:
            orbit_style = orbit_style.copy() # so we don't mess with mutable objects
        for k,v in default_orbit_style.items():
            if k not in orbit_style:
                orbit_style[k] = v

        w_coord = galactocentric_frame.realize_frame(coord.CartesianRepresentation(orbit_w.T[:3]*u.kpc))\
                                      .transform_to(coord.Galactic)
        w_oph = w_coord.transform_to(Ophiuchus)
        w_vel = gc.vgal_to_hel(w_coord, orbit_w.T[3:]*u.kpc/u.Myr,
                               galactocentric_frame=galactocentric_frame,
                               vcirc=vcirc, vlsr=vlsr)

        if use_stream_coords:
            x = w_oph.phi1.wrap_at(180*u.deg).to(xlim.unit).value
            y = w_oph.phi2.to(ylim.unit).value
        else:
            x = w_coord.l.to(xlim.unit).value
            y = w_coord.b.to(ylim.unit).value

        axes[0].plot(x, y, **orbit_style)
        axes[1].plot(x, w_coord.distance.to(lims['distance'].unit).value, **orbit_style)

        for i,k in enumerate(ophdata.veloc.keys()):
            this_lims = lims[k]
            axes[i+2].plot(x, w_vel[i].to(this_lims.unit).value, **orbit_style)

    # bottom axis label
    axes[2].set_xlabel(xlabel)
    axes[3].set_xlabel(xlabel)
    axes[4].set_xlabel(xlabel)

    # vertical axis labels
    axes[0].set_ylabel(yylabel)
    axes[1].set_ylabel(r'$d$ [kpc]')
    axes[2].set_ylabel(r'$\mu_l$ [mas yr$^{-1}$]')
    axes[3].set_ylabel(r'$\mu_b$ [mas yr$^{-1}$]')
    axes[4].set_ylabel(r'$v_{\rm los}$ [km s$^{-1}$]')

    # set all phi1 lims
    axes[0].set_xlim(*xlim.value)
    fig.tight_layout()
    axes[-1].set_visible(False) # HACK because matplotlib bbox issues with invisible plots

    return fig
