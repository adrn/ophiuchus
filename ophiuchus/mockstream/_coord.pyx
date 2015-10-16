# coding: utf-8
# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Coordinate help for generating mock streams. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

from libc.math cimport M_PI

cdef extern from "math.h":
    double sqrt(double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil
    double atan2(double y, double x) nogil
    double fmod(double y, double x) nogil

cdef void sat_rotation_matrix(double *w, # in
                              double *R): # out
    cdef:
        double x1_norm, x2_norm, x3_norm = 0.
        unsigned int i
        double *x1 = [0.,0.,0.]
        double *x2 = [0.,0.,0.]
        double *x3 = [0.,0.,0.]

    x1[0] = w[0]
    x1[1] = w[1]
    x1[2] = w[2]

    x3[0] = x1[1]*w[2+3] - x1[2]*w[1+3]
    x3[1] = x1[2]*w[0+3] - x1[0]*w[2+3]
    x3[2] = x1[0]*w[1+3] - x1[1]*w[0+3]

    x2[0] = -x1[1]*x3[2] + x1[2]*x3[1]
    x2[1] = -x1[2]*x3[0] + x1[0]*x3[2]
    x2[2] = -x1[0]*x3[1] + x1[1]*x3[0]

    x1_norm = sqrt(x1[0]*x1[0] + x1[1]*x1[1] + x1[2]*x1[2])
    x2_norm = sqrt(x2[0]*x2[0] + x2[1]*x2[1] + x2[2]*x2[2])
    x3_norm = sqrt(x3[0]*x3[0] + x3[1]*x3[1] + x3[2]*x3[2])

    for i in range(3):
        x1[i] /= x1_norm
        x2[i] /= x2_norm
        x3[i] /= x3_norm

    R[0] = x1[0]
    R[1] = x1[1]
    R[2] = x1[2]
    R[3] = x2[0]
    R[4] = x2[1]
    R[5] = x2[2]
    R[6] = x3[0]
    R[7] = x3[1]
    R[8] = x3[2]

cdef void to_sat_coords(double *w, double *w_sat, double *R, # in
                        double *w_prime): # out
    # Translate to be centered on progenitor
    cdef:
        double *dw = [0.,0.,0.,0.,0.,0.]
        int i

    for i in range(6):
        dw[i] = w[i] - w_sat[i]

    # Project into new basis
    w_prime[0] = dw[0]*R[0] + dw[1]*R[1] + dw[2]*R[2]
    w_prime[1] = dw[0]*R[3] + dw[1]*R[4] + dw[2]*R[5]
    w_prime[2] = dw[0]*R[6] + dw[1]*R[7] + dw[2]*R[8]

    w_prime[3] = dw[3]*R[0] + dw[4]*R[1] + dw[5]*R[2]
    w_prime[4] = dw[3]*R[3] + dw[4]*R[4] + dw[5]*R[5]
    w_prime[5] = dw[3]*R[6] + dw[4]*R[7] + dw[5]*R[8]

cdef void from_sat_coords(double *w_prime, double *w_sat, double *R, # in
                          double *w): # out
    cdef int i

    # Project back from sat plane
    w[0] = w_prime[0]*R[0] + w_prime[1]*R[3] + w_prime[2]*R[6]
    w[1] = w_prime[0]*R[1] + w_prime[1]*R[4] + w_prime[2]*R[7]
    w[2] = w_prime[0]*R[2] + w_prime[1]*R[5] + w_prime[2]*R[8]

    w[3] = w_prime[3]*R[0] + w_prime[4]*R[3] + w_prime[5]*R[6]
    w[4] = w_prime[3]*R[1] + w_prime[4]*R[4] + w_prime[5]*R[7]
    w[5] = w_prime[3]*R[2] + w_prime[4]*R[5] + w_prime[5]*R[8]

    for i in range(6):
        w[i] = w[i] + w_sat[i]

# ---------------------------------------------------------------------

cdef void car_to_cyl(double *xyz, # in
                     double *cyl): # out
    cdef:
        double R = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1])
        double phi = atan2(xyz[1], xyz[0])
        double z = xyz[2]

    cyl[0] = R
    if (phi >= 0):
        cyl[1] = phi
    else:
        cyl[1] = phi + 2*M_PI
    cyl[2] = z

cdef void cyl_to_car(double *cyl, # in
                     double *xyz): # out
    xyz[0] = cyl[0] * cos(cyl[1])
    xyz[1] = cyl[0] * sin(cyl[1])
    xyz[2] = cyl[2]

cdef void v_car_to_cyl(double *xyz, double *vxyz, double *cyl, # in
                       double *vcyl): # out
    cdef:
        double vR = (xyz[0]*vxyz[0] + xyz[1]*vxyz[1]) / cyl[0]
        double vphi = (xyz[0]*vxyz[1] - vxyz[0]*xyz[1]) / cyl[0]

    vcyl[0] = vR
    vcyl[1] = vphi
    vcyl[2] = vxyz[2]

cdef void v_cyl_to_car(double *cyl, double *vcyl, double *xyz, # in
                       double *vxyz): # out
    vxyz[0] = vcyl[0] * xyz[0]/cyl[0] - vcyl[1] * xyz[1]/cyl[0]
    vxyz[1] = vcyl[0] * xyz[1]/cyl[0] + vcyl[1] * xyz[0]/cyl[0]
    vxyz[2] = vcyl[2]

# ---------------------------------------------------------------------
# Tests
#

cpdef _test_sat_rotation_matrix():
    import numpy as np
    n = 1024

    cdef:
        double[::1] w = np.zeros(6)
        double[::1] wrot = np.zeros(6)
        double[::1] w2 = np.zeros(6)
        double[:,::1] R = np.zeros((3,3))

    for i in range(n):
        w = np.random.uniform(size=6)
        sat_rotation_matrix(&w[0], &R[0,0])

        x = np.array(R).dot(np.array(w)[:3])
        assert x[0] > 0
        assert np.allclose(x[1], 0)
        assert np.allclose(x[2], 0)

        v = np.array(R).dot(np.array(w)[3:])
        assert np.allclose(v[2], 0)
        for j in range(3):
            wrot[j] = x[j]
            wrot[j+3] = v[j]

        x2 = np.array(R.T).dot(np.array(wrot)[:3])
        v2 = np.array(R.T).dot(np.array(wrot)[3:])
        for j in range(3):
            w2[j] = x2[j]
            w2[j+3] = v2[j]

        for j in range(6):
            assert np.allclose(w[j], w2[j])

cpdef _test_car_to_cyl_roundtrip():
    import numpy as np
    n = 1024

    cdef:
        double[:,::1] xyz = np.random.uniform(-10,10,size=(n,3))
        double[::1] cyl = np.zeros(3)
        double[::1] xyz2 = np.zeros(3)

    for i in range(n):
        car_to_cyl(&xyz[i,0], &cyl[0])
        cyl_to_car(&cyl[0], &xyz2[0])
        for j in range(3):
            assert np.allclose(xyz[i,j], xyz2[j])

cpdef _test_cyl_to_car_roundtrip():
    import numpy as np
    n = 1024

    cdef:
        double[:,::1] cyl = np.random.uniform(0,2*np.pi,size=(n,3))
        double[::1] xyz = np.zeros(3)
        double[::1] cyl2 = np.zeros(3)

    for i in range(n):
        cyl_to_car(&cyl[i,0], &xyz[0])
        car_to_cyl(&xyz[0], &cyl2[0])
        for j in range(3):
            assert np.allclose(cyl[i,j], cyl2[j])

cpdef _test_vcar_to_cyl_roundtrip():
    import numpy as np
    n = 1024

    cdef:
        double[:,::1] xyz = np.random.uniform(-10,10,size=(n,3))
        double[:,::1] vxyz = np.random.uniform(-10,10,size=(n,3))
        double[::1] cyl = np.zeros(3)
        double[::1] vcyl = np.zeros(3)
        double[::1] xyz2 = np.zeros(3)
        double[::1] vxyz2 = np.zeros(3)

    for i in range(n):
        car_to_cyl(&xyz[i,0], &cyl[0])
        v_car_to_cyl(&xyz[i,0], &vxyz[i,0], &cyl[0], &vcyl[0])
        v_cyl_to_car(&cyl[0], &vcyl[0], &xyz[i,0], &vxyz2[0])
        for j in range(3):
            assert np.allclose(vxyz[i,j], vxyz2[j])

cpdef _test_vcyl_to_car_roundtrip():
    import numpy as np
    n = 1024

    cdef:
        double[:,::1] cyl = np.random.uniform(0,2*np.pi,size=(n,3))
        double[:,::1] vcyl = np.random.uniform(-10,10,size=(n,3))
        double[::1] xyz = np.zeros(3)
        double[::1] vxyz = np.zeros(3)
        double[::1] vcyl2 = np.zeros(3)

    for i in range(n):
        cyl_to_car(&cyl[i,0], &xyz[0])
        v_cyl_to_car(&cyl[i,0], &vcyl[i,0], &xyz[0], &vxyz[0])
        v_car_to_cyl(&xyz[0], &vxyz[0], &cyl[i,0], &vcyl2[0])
        for j in range(3):
            assert np.allclose(vcyl[i,j], vcyl2[j])
