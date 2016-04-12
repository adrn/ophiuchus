#include <math.h>
#include <string.h>
#include "bfe.h"

double wang_zhao_coeff[392] = {1.509, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.606, 0.0, 0.665, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.406, 0.0, -0.66, 0.0, 0.044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.859, 0.0, 0.984, 0.0, -0.03, 0.0, 0.001, -0.086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.221, 0.0, 0.129, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.295, 0.0, -0.14, 0.0, -0.012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.033, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.001, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

double wang_zhao_bar_value(double t, double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (length scale)
            - initial bar angle
            - pattern speed
    */

    double bar_angle0 = pars[3];
    double pattern_speed = -pars[4]; // added minus sign to make it rotate correctly
    double alpha = (-bar_angle0 + pattern_speed*t); // % (2 * 3.141592653589793238462643383));

    double rot_r[3];
    double new_pars[5+392];

    double cosa = cos(alpha);
    double sina = sin(alpha);
    rot_r[0] = cosa*r[0] + sina*r[1];
    rot_r[1] = -sina*r[0] + cosa*r[1];
    rot_r[2] = r[2];

    new_pars[0] = pars[0];
    new_pars[1] = pars[1];
    new_pars[2] = pars[2];
    new_pars[3] = 3.; // nmax
    new_pars[4] = 6.; // lmax
    for (int i=0; i<392; i++) {
        new_pars[5+i] = wang_zhao_coeff[i];
    }
    return scf_value(t, &new_pars[0], &rot_r[0]);
}

void wang_zhao_bar_gradient(double t, double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (length scale)
            - initial bar angle
            - pattern speed
    */

    double bar_angle0 = pars[3];
    double pattern_speed = -pars[4]; // added minus sign to make it rotate correctly
    double alpha = (-bar_angle0 + pattern_speed*t); // % (2 * 3.141592653589793238462643383));

    // printf("%.5f %.5f %.5f %.5f %.5f\n", alpha, pattern_speed, pars[0], pars[1], pars[2]);

    double rot_r[3];
    double new_pars[5+392];

    double tmp1,tmp2;

    double cosa = cos(alpha);
    double sina = sin(alpha);
    rot_r[0] = cosa*r[0] + sina*r[1];
    rot_r[1] = -sina*r[0] + cosa*r[1];
    rot_r[2] = r[2];

    new_pars[0] = pars[0];
    new_pars[1] = pars[1];
    new_pars[2] = pars[2];
    new_pars[3] = 3.; // nmax
    new_pars[4] = 6.; // lmax
    for (int i=0; i<392; i++) {
        new_pars[5+i] = wang_zhao_coeff[i];
    }
    scf_gradient(t, &new_pars[0], &rot_r[0], &grad[0]);

    tmp1 = cosa*grad[0] - sina*grad[1];
    tmp2 = sina*grad[0] + cosa*grad[1];
    grad[0] = tmp1;
    grad[1] = tmp2;
}

double wang_zhao_bar_density(double t, double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (length scale)
            - initial bar angle
            - pattern speed
    */

    double bar_angle0 = pars[3];
    double pattern_speed = -pars[4]; // added minus sign to make it rotate correctly
    double alpha = (-bar_angle0 + pattern_speed*t); // % (2 * 3.141592653589793238462643383));

    double rot_r[3];
    double new_pars[5+392];

    double cosa = cos(alpha);
    double sina = sin(alpha);
    rot_r[0] = cosa*r[0] + sina*r[1];
    rot_r[1] = -sina*r[0] + cosa*r[1];
    rot_r[2] = r[2];

    new_pars[0] = pars[0];
    new_pars[1] = pars[1];
    new_pars[2] = pars[2];
    new_pars[3] = 3.; // nmax
    new_pars[4] = 6.; // lmax
    for (int i=0; i<392; i++) {
        new_pars[5+i] = wang_zhao_coeff[i];
    }
    return scf_density(t, &new_pars[0], &rot_r[0]);
}

// double ophiuchus_value(double t, double *pars, double*r) {
//     double v = 0.;

//     if (pars[1] > 0) {
//         v += hernquist_value(0., &pars[0], &r[0]);
//     }
//     if (pars[4] > 0) {
//         v += miyamotonagai_value(0., &pars[3], &r[0]);
//     }
//     if (pars[8] > 0) {
//         v += flattenednfw_value(0., &pars[7], &r[0]);
//     }
//     if (pars[12] > 0) {
//         v += wang_zhao_bar_value(t, &pars[11], &r[0]);
//     }
//     return v;
// }

// void ophiuchus_gradient(double t, double *pars, double *r, double *grad) {
//     double tmp_grad[3];
//     int i;

//     for (i=0; i<3; i++) grad[i] = 0;

//     if (pars[1] > 0) {
//         hernquist_gradient(0., &pars[0], &r[0], &tmp_grad[0]);
//         for (i=0; i<3; i++) grad[i] += tmp_grad[i];
//     }

//     if (pars[4] > 0) {
//         miyamotonagai_gradient(0., &pars[3], &r[0], &tmp_grad[0]);
//         for (i=0; i<3; i++) grad[i] += tmp_grad[i];
//     }

//     if (pars[8] > 0) {
//         flattenednfw_gradient(0., &pars[7], &r[0], &tmp_grad[0]);
//         for (i=0; i<3; i++) grad[i] += tmp_grad[i];
//     }

//     if (pars[12] > 0) {
//         wang_zhao_bar_gradient(t, &pars[11], &r[0], &tmp_grad[0]);
//         for (i=0; i<3; i++) grad[i] += tmp_grad[i];
//     }
// }

// double ophiuchus_density(double t, double *pars, double*r) {
//     double v = 0.;

//     if (pars[1] > 0) {
//         v += hernquist_density(0., &pars[0], &r[0]);
//     }
//     if (pars[4] > 0) {
//         v += miyamotonagai_density(0., &pars[3], &r[0]);
//     }
//     if (pars[8] > 0) {
//         v += flattenednfw_density(0., &pars[7], &r[0]);
//     }
//     if (pars[12] > 0) {
//         v += wang_zhao_bar_density(t, &pars[11], &r[0]);
//     }
//     return v;
// }
