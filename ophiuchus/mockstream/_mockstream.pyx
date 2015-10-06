# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Generate mock streams. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

from gary.potential.cpotential cimport _CPotential

cdef extern from "math.h":
    double sqrt(double x) nogil
    double log(double x) nogil
    double atan2(double y, double x) nogil
    double acos(double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil

cdef extern from "dop853.h":
    ctypedef void (*GradFn)(double *pars, double *q, double *grad) nogil
    ctypedef void (*SolTrait)(long nr, double xold, double x, double* y, unsigned n, int* irtrn)
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f, GradFn gradfunc, double *gpars, unsigned norbits) nogil

    # See dop853.h for full description of all input parameters
    int dop853 (unsigned n, FcnEqDiff fcn, GradFn gradfunc, double *gpars, unsigned norbits,
                double x, double* y, double xend,
                double* rtoler, double* atoler, int itoler, SolTrait solout,
                int iout, FILE* fileout, double uround, double safe, double fac1,
                double fac2, double beta, double hmax, double h, long nmax, int meth,
                long nstiff, unsigned nrdens, unsigned* icont, unsigned licont)

    void Fwrapper (unsigned ndim, double t, double *w, double *f,
                   GradFn func, double *pars, unsigned norbits)
    double six_norm (double *x)

cdef extern from "stdio.h":
    ctypedef struct FILE
    FILE *stdout

cdef void car_to_sph(double *xyz, double *sph):
    # TODO: note this isn't consistent with the velocity transform because of theta
    # get out spherical components
    cdef:
        double d = sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2])
        double phi = atan2(xyz[1], xyz[0])
        double theta = acos(xyz[2] / d)

    sph[0] = d
    sph[1] = phi
    sph[2] = theta

cdef void sph_to_car(double *sph, double *xyz):
    # TODO: note this isn't consistent with the velocity transform because of theta
    # get out spherical components
    xyz[0] = sph[0] * cos(sph[1]) * sin(sph[2])
    xyz[1] = sph[0] * sin(sph[1]) * sin(sph[2])
    xyz[2] = sph[0] * cos(sph[2])

cdef void v_car_to_sph(double *xyz, double *vxyz, double *vsph):
    # get out spherical components
    cdef:
        double d = sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2])
        double dxy = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1])

        double vr = (xyz[0]*vxyz[0]+xyz[1]*vxyz[1]+xyz[2]*vxyz[2]) / d

        double mu_lon = (xyz[0]*vxyz[1] - vxyz[0]*xyz[1]) / (dxy*dxy)
        double vlon = mu_lon * dxy # cos(lat)

        double mu_lat = (xyz[2]*(xyz[0]*vxyz[0] + xyz[1]*vxyz[1]) - dxy*dxy*vxyz[2]) / (d*d*dxy)
        double vlat = -mu_lat * d

    vsph[0] = vr
    vsph[1] = vlon
    vsph[2] = vlat

cdef void v_sph_to_car(double *xyz, double *vsph, double *vxyz):
    # get out spherical components
    cdef:
        double d = sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2])
        double dxy = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1])

    vxyz[0] = vsph[0]*xyz[0]/dxy*dxy/d - xyz[1]/dxy*vsph[1] - xyz[0]/dxy*xyz[2]/d*vsph[2]
    vxyz[1] = vsph[0]*xyz[1]/dxy*dxy/d + xyz[0]/dxy*vsph[1] - xyz[1]/dxy*xyz[2]/d*vsph[2]
    vxyz[2] = vsph[0]*xyz[2]/d + dxy/d*vsph[2]

cpdef make_stream(_CPotential cpotential, double[::1] t, double[:,::1] prog_w,
                  int release_every, double G, double prog_mass,
                  double rscale, double vscale,
                  double atol, double rtol):
    """
    generate_stream(cpotential, t, prog_w, release_every, G, prog_mass, rscale, vscale, atol, rtol)
    """
    cdef:
        int i, j, k
        int res
        int nsteps = t.shape[0]
        unsigned norbits = 2 * nsteps / release_every
        unsigned ndim = prog_w.shape[1]
        unsigned ndim_2 = ndim / 2
        double[::1] w = np.empty(norbits*ndim)
        double dt0 = t[1] - t[0]
        double[::1] tmpv = np.zeros(3)
        double sigmar, sigmav

        unsigned this_ndim, this_norbits

        double r_tide

        # ignore this
        double[::1] eps = np.zeros(3)

    # store initial conditions
    i = 0
    for j in range(nsteps-1):
        if (j % release_every) != 0:
            continue

        for k in range(ndim):
            w[2*i*ndim + k] = prog_w[j,k]
            w[2*i*ndim + k + ndim] = prog_w[j,k]

        i += 1

    # TODO: now go back and add scatter, tidal radius offset, etc.
    i = 0
    for j in range(nsteps-1):
        if (j % release_every) != 0:
            continue

        menc = cpotential._mass_enclosed(t[j], &prog_w[j,0], &eps[0], G)
        sigmar = rscale * (prog_mass / menc)**(1/3.) * \
                  sqrt(prog_w[j,0]**2 + prog_w[j,1]**2 + prog_w[j,2]**2) / 2.
        sigmav = vscale * (prog_mass / menc)**(1/3.) * \
                  sqrt(prog_w[j,3]**2 + prog_w[j,4]**2 + prog_w[j,5]**2) / 2.

        # Gaussian spheres in position, offset in radial dir, scatter in all
        car_to_sph(&w[2*i*ndim], &tmpv[0])
        tmpv[0] = tmpv[0] + sigmar
        sph_to_car(&tmpv[0], &w[2*i*ndim])

        car_to_sph(&w[2*i*ndim + ndim], &tmpv[0])
        tmpv[0] = tmpv[0] - sigmar
        sph_to_car(&tmpv[0], &w[2*i*ndim + ndim])

        for k in range(ndim_2):
            w[2*i*ndim + k] = np.random.normal(w[2*i*ndim + k], sigmar / 1.732 / 2.)
            w[2*i*ndim + k + ndim] = np.random.normal(w[2*i*ndim + k + ndim], sigmar / 1.732 / 2.)

        # -------- velocity --------
        # add scatter in radial velocity only
        v_car_to_sph(&w[2*i*ndim], &w[2*i*ndim + 3], &tmpv[0])
        tmpv[0] = np.random.normal(tmpv[0], sigmav)
        v_sph_to_car(&w[2*i*ndim], &tmpv[0], &w[2*i*ndim + 3])

        v_car_to_sph(&w[2*i*ndim + ndim], &w[2*i*ndim + ndim + 3], &tmpv[0])
        tmpv[0] = np.random.normal(tmpv[0], sigmav)
        v_sph_to_car(&w[2*i*ndim + ndim], &tmpv[0], &w[2*i*ndim + ndim + 3])

        i += 1

    # define full array of times
    i = 1
    for j in range(0,nsteps-1,1):
        if j % release_every == 0:
            this_norbits = 2*i
            this_ndim = ndim * this_norbits
            i += 1

        res = dop853(this_ndim, <FcnEqDiff> Fwrapper,
                     <GradFn>cpotential.c_gradient, &(cpotential._parameters[0]), this_norbits,
                     t[j], &w[0], t[j+1], &rtol, &atol, 0, NULL, 0,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt0, 0, 0, 1, 0, NULL, 0);

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stff (interrupted).")

    return np.asarray(w).reshape(norbits, ndim)
