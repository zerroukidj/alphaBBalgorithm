#define _USE_MATH_DEFINES
#include <iostream>
#include<stdio.h>
#include<math.h>
#include <boost/numeric/interval.hpp>
#include "multi_dimensional_root_finding.hpp"
#include<chrono>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <iomanip>
#include <vector>
#include <nlopt.h>

using Eigen::MatrixXd;

using namespace boost;
using namespace numeric;
using namespace interval_lib;


double k[5], k_v[5], sig[5], Kq, ksg[5], gamma[5], sigma[5], K1, L, U;
double pii = 3.14, timing;
double glb[5], sbar[5], sbark[5][3], sopt[5];
double UB, LB, LBk[10000][3], xl[10000][5];
int n = 2, Ni = 4, convex;
typedef interval<double,
    policies<rounded_math<double>,
    checking_base<double> > > I;

I xg[5], Tg[5], Tsg[5];
I T[5], Tk[5][3], Ts[5];

double calc_sigma() {

    double gam;

    const size_t systemSize1 = 2;
    //equation dim1
    NonlinearSystem<systemSize1> sys0;
    equation_type eq1 = [](const vector_type& x) {

        return -(K1) + pow(x[0], 2) + pow(x[0], 2) * exp(x[0] * (U - L));
        };
    sys0.assign_equation(eq1, 0);

    equation_type eq2 = [](const vector_type& x) {
        return -(K1) + pow(x[1], 2) + pow(x[1], 2) * exp(x[1] * (U - L));
        };
    sys0.assign_equation(eq2, 1);
    array<double, systemSize1> init0 = { 1, 1 };
    sys0.initialize(init0);
    sys0.find_roots_gnewton();
    gam = sys0.argument(0);

    return gam;
}

  /* Hessian interval matrices
  
  f1
         H[1][1] = sin_(x[1] * x[2]) * sin_(x[1]) - I(2) * x[2] * cos_(x[1] * x[2]) * cos_(x[1]) + pow(x[2], 2) * sin_(x[1] * x[2]) * sin_(x[1]);
         H[1][2] = x[1] * x[2] * sin_(x[1] * x[2]) * sin_(x[1]) - x[1] * cos_(x[1] * x[2]) * cos_(x[1]) - cos_(x[1] * x[2]) * sin_(x[1]);
         H[2][1] = x[1] * x[2] * sin_(x[1] * x[2]) * sin_(x[1]) - x[1] * cos_(x[1] * x[2]) * cos_(x[1]) - cos_(x[1] * x[2]) * sin_(x[1]);
         H[2][2] = pow(x[1], 2) * sin_(x[1] * x[2]) * sin_(x[1]);

     */

     /* f2
     H[1][1] = (I(4) * sin_(I(2) * x[1] + x[2])) / (sin_(x[2]) + I(2));
     H[1][2] = (I(2) * sin_(I(2) * x[1] + x[2])) / (sin_(x[2]) + I(2)) + (I(2) * cos_(I(2) * x[1] + x[2]) * cos_(x[2])) / pow(sin_(x[2]) + I(2), 2);
     H[2][1] = (I(2) * sin_(I(2) * x[1] + x[2])) / (sin_(x[2]) + I(2)) + (I(2) * cos_(I(2) * x[1] + x[2]) * cos_(x[2])) / pow(sin_(x[2]) + I(2), 2);
     H[2][2] = sin_(I(2) * x[1] + x[2]) / (sin_(x[2]) + I(2)) - (sin_(I(2) * x[1] + x[2]) * sin_(x[2])) / pow(sin_(x[2]) + I(2), 2) - (I(2) * sin_(I(2) * x[1] + x[2]) * pow(cos_(x[2]), 2)) / pow(sin_(x[2]) + I(2), 3) + (I(2) * cos_(I(2) * x[1] + x[2]) * cos_(x[2])) / pow(sin_(x[2]) + I(2), 2);
     */

     /* f3
  H[1][1] = I(2) - sin_(x[1] + x[2]);
  H[1][2] = -sin_(x[1] + x[2]) - I(2);
  H[2][1] = -sin_(x[1] + x[2]) - I(2);
  H[2][2] = I(2) - sin_(x[1] + x[2]);
  */

  /*f4
      H[1][1] = sin_((x[1] - I(1)) * (x[1] - I(2)) * (x[2] + I(1))) * pow((x[1] - I(1)) * (x[2] + I(1)) + (x[1] - I(2)) * (x[2] + I(1)), 2) - cos_((x[1] - I(1)) * (x[1] - I(2)) * (x[2] + I(1))) * (I(2) * x[2] + I(2));
      H[1][2] = sin_((x[1] - I(1)) * (x[1] - I(2)) * (x[2] + I(1))) * ((x[1] - I(1)) * (x[2] + I(1)) + (x[1] - I(2)) * (x[2] + I(1))) * (x[1] - I(1)) * (x[1] - I(2)) - cos_((x[1] - I(1)) * (x[1] - I(2)) * (x[2] + I(1))) * (x[1] - I(2)) - cos_((x[1] - I(1)) * (x[1] - I(2)) * (x[2] + I(1))) * (x[1] - I(1));
      H[2][1] = sin_((x[1] - I(1)) * (x[1] - I(2)) * (x[2] + I(1))) * ((x[1] - I(1)) * (x[2] + I(1)) + (x[1] - I(2)) * (x[2] + I(1))) * (x[1] - I(1)) * (x[1] - I(2)) - cos_((x[1] - I(1)) * (x[1] - I(2)) * (x[2] + I(1))) * (x[1] - I(2)) - cos_((x[1] - I(1)) * (x[1] - I(2)) * (x[2] + I(1))) * (x[1] - I(1));
      H[2][2] = sin_((x[1] - I(1)) * (x[1] - I(2)) * (x[2] + I(1))) * pow(x[1] - I(1), 2) * pow(x[1] - I(2), 2);
      */

      /*f5
          H[1][1] = I(1) / (I(50) * pow(pow(x[1], 2) / I(4) + pow(x[2], 2) - I(1), 2)) - pow(x[1], 2) / (I(50) * pow(pow(x[1], 2) / I(4) + pow(x[2], 2) - I(1), 3)) + I(12);
          H[1][2] = -(I(2) * x[1] * x[2]) / (I(25) * pow(pow(x[1], 2) / I(4) + pow(x[2], 2) - I(1), 3)) - I(20);
          H[2][1] = -(I(2) * x[1] * x[2]) / (I(25) * pow(pow(x[1], 2) / I(4) + pow(x[2], 2) - I(1), 3)) - I(20);
          H[2][2] = I(2) / (I(25) * pow(pow(x[1], 2) / I(4) + pow(x[2], 2) - I(1), 2)) - (I(8) * pow(x[2], 2)) / (I(25) * pow(pow(x[1], 2) / I(4) + pow(x[2], 2) - I(1), 3)) + I(42);

      */

      /* f6
          H[1][1] = I(2) * pow((I(6250) * x[1]) / I(24649) - I(250 / 157), 2) - (I(12500) * x[2]) / I(24649) - (I(3015) * cos_(x[1])) / I(314) - (I(3125000) * x[1]) / I(3869893) + (I(39062500) * pow(x[1], 2)) / I(607573201) + I(75000 / 24649);
          H[1][2] = I(500 / 157) - (I(12500) * x[1]) / I(24649);
          H[2][1] = I(500 / 157) - (I(12500) * x[1]) / I(24649);
          H[2][2] = I(2);

      */

      /*    f7
          H[1][1] = I(1200) * pow(x[1], 2) - I(400) * x[2] + I(2);
          H[1][2] = I(-400) * x[1];
          H[2][1] = I(-400) * x[1];
          H[2][2] = I(200);

      */

      /*    f8
          H[1][1] = (I(3) * (pow(x[2], 2) + I(1))) / (I(5) * pow(x[1], 4)) - I(14) / (pow(x[1], 4) * pow(x[2], 2)) + (I(20) * (pow(x[1], 2) * pow(x[2], 2) + I(100))) / (pow(x[1], 6) * pow(x[2], 4)) + I(1 / 5);
          H[1][2] = (I(16) * (pow(x[1], 2) * pow(x[2], 2) + I(100))) / (pow(x[1], 5) * pow(x[2], 5)) - (I(2) * x[2]) / (I(5) * pow(x[1], 3)) - I(12) / (pow(x[1], 3) * pow(x[2], 3));
          H[2][1] = (I(16) * (pow(x[1], 2) * pow(x[2], 2) + I(100))) / (pow(x[1], 5) * pow(x[2], 5)) - (I(2) * x[2]) / (I(5) * pow(x[1], 3)) - I(12) / (pow(x[1], 3) * pow(x[2], 3));
          H[2][2] = I(1) / (I(5) * pow(x[1], 2)) - I(14) / (pow(x[1], 2) * pow(x[2], 4)) + (I(20) * (pow(x[1], 2) * pow(x[2], 2) + I(100))) / (pow(x[1], 4) * pow(x[2], 6));

      */
      /*    f9
          H[1][1] = (I(100) * cos_(I(10) * log_(I(2) * x[1])) * cos_(I(10) * log_(I(3) * x[2]))) / pow(x[1], 2) - (I(10) * cos_(I(10) * log_(I(3) * x[2])) * sin_(I(10) * log_(I(2) * x[1]))) / pow(x[1], 2) + I(1);
          H[1][2] = -(I(100) * sin_(I(10) * log_(I(2) * x[1])) * sin_(I(10) * log_(I(3) * x[2]))) / (x[1] * x[2]);
          H[2][1] = -(I(100) * sin_(I(10) * log_(I(2) * x[1])) * sin_(I(10) * log_(I(3) * x[2]))) / (x[1] * x[2]);
          H[2][2] = (I(100) * cos_(I(10) * log_(I(2) * x[1])) * cos_(I(10) * log_(I(3) * x[2]))) / pow(x[2], 2) - (I(10) * cos_(I(10) * log_(I(2) * x[1])) * sin_(I(10) * log_(I(3) * x[2]))) / pow(x[2], 2) + I(1);

      */
      /*    f10
          H[1][1] = I(2) * ((I(8) * x[1] - I(12) * x[2]) * (I(12) * pow(x[1], 2) - I(36) * x[1] * x[2] - I(32) * x[1] + I(27) * pow(x[2], 2) + I(48) * x[2] + I(18)) - pow(I(2) * x[1] - I(3) * x[2], 2) * (I(36) * x[2] - I(24) * x[2] + I(32))) * ((I(6) * x[1] + I(6) * x[2] - I(14)) * pow(x[1] + x[2] + I(1), 2) + (I(2) * x[1] + I(2) * x[2] + I(2)) * (I(3) * pow(x[1], 2) + I(6) * x[1] * x[2] - I(14) * x[1] + I(3) * pow(x[2], 2) - I(14) * x[2] + I(19))) + (pow(I(2) * x[1] - I(3) * x[2], 2) * (I(12) * pow(x[1], 2) - I(36) * x[1] * x[2] - I(32) * x[1] + I(27) * pow(x[2], 2) + I(48) * x[2] + I(18)) + I(30)) * (I(12) * x[1] * x[2] - I(28) * x[2] - I(28) * x[1] + I(2) * (I(2) * x[1] + I(2) * x[2] + I(2)) * (I(6) * x[1] + I(6) * x[2] - I(14)) + I(6) * pow(x[1] + x[2] + I(1), 2) + I(6) * pow(x[1], 2) + I(6) * pow(x[2], 2) + I(38)) + (pow(x[1] + x[2] + I(1), 2) * (I(3) * pow(x[1], 2) + I(6) * x[1] * x[2] - I(14) * x[1] + I(3) * pow(x[2], 2) - I(14) * x[2] + I(19)) + I(1)) * (I(384) * x[2] - I(256) * x[1] - I(288) * x[1] * x[2] + I(24) * pow(I(2) * x[1] - I(3) * x[2], 2) - I(2) * (I(8) * x[1] - I(12) * x[2]) * (I(36) * x[2] - I(24) * x[1] + I(32)) + I(96) * pow(x[1], 2) + I(216) * pow(x[2], 2) + I(144));
          H[1][2] = ((I(8) * x[1] - I(12) * x[2]) * (I(12) * pow(x[1], 2) - I(36) * x[1] * x[2] - I(32) * x[1] + I(27) * pow(x[2], 2) + I(48) * x[2] + I(18)) - pow(I(2) * x[1] - I(3) * x[2], 2) * (I(36) * x[2] - I(24) * x[1] + I(32))) * ((I(6) * x[1] + I(6) * x[2] - I(14)) * pow(x[1] + x[2] + I(1), 2) + (I(2) * x[1] + I(2) * x[2] + I(2)) * (I(3) * pow(x[1], 2) + I(6) * x[1] * x[2] - I(14) * x[1] + I(3) * pow(x[2], 2) - I(14) * x[2] + I(19))) - (pow(x[1] + x[2] + I(1), 2) * (I(3) * pow(x[1], 2) + I(6) * x[1] * x[2] - I(14) * x[1] + I(3) * pow(x[2], 2) - I(14) * x[2] + I(19)) + I(1)) * (I(576) * x[2] - I(384) * x[1] - I(432) * x[1] * x[2] + I(36) * pow(I(2) * x[1] - I(3) * x[2], 2) - (I(12) * x[1] - I(18) * x[2]) * (I(36) * x[2] - I(24) * x[1] + I(32)) - (I(8) * x[1] - I(12) * x[2]) * (I(54) * x[2] - I(36) * x[1] + I(48)) + I(144) * pow(x[1], 2) + I(324) * pow(x[2], 2) + I(216)) - ((I(12) * x[1] - I(18) * x[2]) * (I(12) * pow(x[1], 2) - I(36) * x[1] * x[2] - I(32) * x[1] + I(27) * pow(x[1], 2) + I(48) * x[2] + I(18)) - pow(I(2) * x[1] - I(3) * x[2], 2) * (I(54) * x[2] - I(36) * x[1] + I(48))) * ((I(6) * x[1] + I(6) * x[2] - I(14)) * pow(x[1] + x[2] + I(1), 2) + (I(2) * x[1] + I(2) * x[2] + I(2)) * (I(3) * pow(x[1], 2) + I(6) * x[1] * x[2] - I(14) * x[1] + I(3) * pow(x[2], 2) - I(14) * x[2] + I(19))) + (pow(I(2) * x[1] - I(3) * x[2], 2) * (I(12) * pow(x[1], 2) - I(36) * x[1] * x[2] - I(32) * x[1] + I(27) * pow(x[2], 2) + I(48) * x[2] + I(18)) + I(30)) * (I(12) * x[1] * x[2] - I(28) * x[2] - I(28) * x[1] + I(2) * (I(2) * x[1] + I(2) * x[2] + I(2)) * (I(6) * x[1] + I(6) * x[2] - I(14)) + I(6) * pow(x[1] + x[2] + I(1), 2) + I(6) * pow(x[1], 2) + I(6) * pow(x[2], 2) + I(38));
          H[2][1] = ((I(8) * x[1] - I(12) * x[2]) * (I(12) * pow(x[1], 2) - I(36) * x[1] * x[2] - I(32) * x[1] + I(27) * pow(x[2], 2) + I(48) * x[2] + I(18)) - pow(I(2) * x[1] - I(3) * x[2], 2) * (I(36) * x[2] - I(24) * x[1] + I(32))) * ((I(6) * x[1] + I(6) * x[2] - I(14)) * pow(x[1] + x[2] + I(1), 2) + (I(2) * x[1] + I(2) * x[2] + I(2)) * (I(3) * pow(x[1], 2) + I(6) * x[1] * x[2] - I(14) * x[1] + I(3) * pow(x[2], 2) - I(14) * x[2] + I(19))) - (pow(x[1] + x[2] + I(1), 2) * (I(3) * pow(x[1], 2) + I(6) * x[1] * x[2] - I(14) * x[1] + I(3) * pow(x[2], 2) - I(14) * x[2] + I(19)) + I(1)) * (I(576) * x[2] - I(384) * x[1] - I(432) * x[1] * x[2] + I(36) * pow(I(2) * x[1] - I(3) * x[2], 2) - (I(12) * x[1] - I(18) * x[2]) * (I(36) * x[2] - I(24) * x[1] + I(32)) - (I(8) * x[1] - I(12) * x[2]) * (I(54) * x[2] - I(36) * x[1] + I(48)) + I(144) * pow(x[1], 2) + I(324) * pow(x[2], 2) + I(216)) - ((I(12) * x[1] - I(18) * x[2]) * (I(12) * pow(x[1], 2) - I(36) * x[1] * x[2] - I(32) * x[1] + I(27) * pow(x[1], 2) + I(48) * x[2] + I(18)) - pow(I(2) * x[1] - I(3) * x[2], 2) * (I(54) * x[2] - I(36) * x[1] + I(48))) * ((I(6) * x[1] + I(6) * x[2] - I(14)) * pow(x[1] + x[2] + I(1), 2) + (I(2) * x[1] + I(2) * x[2] + I(2)) * (I(3) * pow(x[1], 2) + I(6) * x[1] * x[2] - I(14) * x[1] + I(3) * pow(x[2], 2) - I(14) * x[2] + I(19))) + (pow(I(2) * x[1] - I(3) * x[2], 2) * (I(12) * pow(x[1], 2) - I(36) * x[1] * x[2] - I(32) * x[1] + I(27) * pow(x[2], 2) + I(48) * x[2] + I(18)) + I(30)) * (I(12) * x[1] * x[2] - I(28) * x[2] - I(28) * x[1] + I(2) * (I(2) * x[1] + I(2) * x[2] + I(2)) * (I(6) * x[1] + I(6) * x[2] - I(14)) + I(6) * pow(x[1] + x[2] + I(1), 2) + I(6) * pow(x[1], 2) + I(6) * pow(x[2], 2) + I(38));
          H[2][2] = (pow(I(2) * x[1] - I(3) * x[2], 2) * (I(12) * pow(x[1], 2) - I(36) * x[1] * x[2] - I(32) * x[1] + I(27) * pow(x[2], 2) + I(48) * x[2] + I(18)) + I(30)) * (I(12) * x[1] * x[2] - I(28) * x[2] - I(28) * x[1] + I(2) * (I(2) * x[1] + I(2) * x[2] + I(2)) * (I(6) * x[1] + I(6) * x[2] - I(14)) + I(6) * pow(x[1] + x[2] + I(1), 2) + I(6) * pow(x[1], 2) + I(6) * pow(x[2], 2) + I(38)) - I(2) * ((I(12) * x[1] - I(18) * x[2]) * (I(12) * pow(x[1], 2) - I(36) * x[1] * x[2] - I(32) * x[1] + I(27) * pow(x[2], 2) + I(48) * x[2] + I(18)) - pow(I(2) * x[1] - I(3) * x[2], 2) * (I(54) * x[2] - I(36) * x[1] + I(48))) * ((I(6) * x[1] + I(6) * x[2] - I(14)) * pow(x[1] + x[2] + I(1), 2) + (I(2) * x[1] + I(2) * x[2] + I(2)) * (I(3) * pow(x[1], 2) + I(6) * x[1] * x[2] - I(14) * x[1] + I(3) * pow(x[2], 2) - I(14) * x[2] + I(19))) + (pow(x[1] + x[2] + I(1), 2) * (I(3) * pow(x[1], 2) + I(6) * x[1] * x[2] - I(14) * x[1] + I(3) * pow(x[2], 2) - I(14) * x[2] + I(19)) + I(1)) * (I(864) * x[2] - I(576) * x[1] - I(648) * x[1] * x[2] + I(54) * pow(I(2) * x[1] - I(3) * x[2], 2) - I(2) * (I(12) * x[1] - I(18) * x[2]) * (I(54) * x[2] - I(36) * x[1] + I(48)) + I(216) * pow(x[1], 2) + I(486) * pow(x[2], 2) + I(324));

      */
      /*
                         f11
                        H[1][1] = I(12) * pow(x[1], 2) - I(2);
                        H[1][2] = I(-4) * x[2];
                        H[2][1] = I(-4) * x[2];
                        H[2][2] = I(-12) * pow(x[2], 2) - I(4) * x[1];

                        f12
          H[1][1] = I(2);
          H[1][2] = -I(2) * exp_(x[2]);
          H[2][1] = -I(2) * exp_(x[2]);
          H[2][2] = I(2) * exp_(I(2) * x[2]) - I(2) * exp_(x[2]) * (x[1] - exp_(x[2]) + I(1));
          f13
          H[1][1] = I(2) * pow(x[2], 2) + I(8);
          H[1][2] = I(4) * x[1] * x[2] + I(2);
          H[2][1] = I(4) * x[1] * x[2] + I(2);
          H[2][2] = I(2) * pow(x[1], 2) + I(2);
          f14
          H[1][1] = I(10) * pow(x[1],4) - ((I(126) * pow(x[1],2)) / I(5)) + I(8);
          H[1][2] = I(1);
          H[2][1] = I(1);
          H[2][2] = I(48) * pow(x[2] ,2) - I(8);
          f15
          H[1][1] = I(12) * pow((x[1] + I(2) * x[2] - I(3)), 2) + I(4);
          H[1][2] = I(24) * pow((x[1] + I(2) * x[2] - I(3)), 2) + I(4);
          H[2][1] = I(24) * pow((x[1] + I(2) * x[2] - I(3)), 2) + I(4);
          H[2][2] = I(48) * pow((x[1] + I(2) * x[2] - I(3)), 2) + I(10);
          f16
          H[1][1] = (I(5) * pow(I(pii), 2) * cos_(I(5) * I(pii) * x[1])) / I(2) + I(2);
          H[1][2] = I(0);
          H[2][1] = I(0);
          H[2][2] = (I(5) * pow(I(pii), 2) * cos_(I(5) * I(pii) * x[2])) / I(2) + I(2);
          */


          //fonction cos
I cos_(I x1) {
    if (x1.upper() - x1.lower() > 2 * M_PI) {
        return I(-1, 1);
    }
    else {
        double  min, max, val1;
        min = cos(x1.lower());
        max = cos(x1.upper());
        for (int i = 0; i <= 10000; i++) {
            val1 = cos(x1.lower() + i * ((x1.upper() - x1.lower()) / 10000));
            if (val1 < min) { min = val1; }
            if (val1 > max) { max = val1; }
        }
        return I(min, max);
    }
}
//function sin
I sin_(I x1) {
    if (x1.upper() - x1.lower() > 2 * M_PI) {
        return I(-1, 1);
    }
    else {
        double  min, max, val1;
        min = sin(x1.lower());
        max = sin(x1.upper());
        for (int i = 0; i <= 10000; i++) {
            val1 = sin(x1.lower() + i * ((x1.upper() - x1.lower()) / 10000));
            if (val1 < min) { min = val1; }
            if (val1 > max) { max = val1; }
        }
        return I(min, max);
    }
}

//fonction tan
I tan_(I x) {
    double a, b;
    if (tan(lower(x)) < tan(upper(x))) {
        a = tan(lower(x));
        b = tan(upper(x));
    }
    if (tan(lower(x)) > tan(upper(x)) || tan(lower(x)) == tan(upper(x))) {
        b = tan(lower(x));
        a = tan(upper(x));
    }

    return I(a, b);
}

//fonction exp
I exp_(I x) {
    double a, b;
    if (exp(lower(x)) < exp(upper(x))) {
        a = exp(lower(x));
        b = exp(upper(x));
    }
    if (exp(lower(x)) > exp(upper(x)) || exp(lower(x)) == exp(upper(x))) {
        b = exp(lower(x));
        a = exp(upper(x));
    }

    return I(a, b);
}

//fonction log
I log_(I x) {
    double a, b;
    if (log(lower(x)) < log(upper(x))) {
        a = log(lower(x));
        b = log(upper(x));
    }
    if (log(lower(x)) > log(upper(x)) || log(lower(x)) == log(upper(x))) {
        b = log(lower(x));
        a = log(upper(x));
    }

    return I(a, b);
}


namespace io_std {

    template<class T, class Policies, class CharType, class CharTraits>
    std::basic_ostream<CharType, CharTraits>& operator<<
        (std::basic_ostream<CharType, CharTraits>& stream,
            const boost::numeric::interval<T, Policies>& value)
    {
        if (empty(value)) {
            return stream << "[]";
        }
        else {
            return stream << '[' << lower(value) << ',' << upper(value) << ']';
        }

    } // namespace io_std

}

void Gershgorin(I x[5], int n) {
    double max, rest, som, dis[5];
    I H[5][5];
    for (int i = 1; i <= n; i++) {
        dis[i] = x[i].upper() - x[i].lower();
        if (dis[i] == 0) { dis[i] = 0.000001; }
        ///std::cout << dis[i] << "  ";
         //dis[i] = 1;
    }
    //f1
    H[1][1] = sin_(x[1] * x[2]) * sin_(x[1]) - I(2) * x[2] * cos_(x[1] * x[2]) * cos_(x[1]) + pow(x[2], 2) * sin_(x[1] * x[2]) * sin_(x[1]);
    H[1][2] = x[1] * x[2] * sin_(x[1] * x[2]) * sin_(x[1]) - x[1] * cos_(x[1] * x[2]) * cos_(x[1]) - cos_(x[1] * x[2]) * sin_(x[1]);
    H[2][1] = x[1] * x[2] * sin_(x[1] * x[2]) * sin_(x[1]) - x[1] * cos_(x[1] * x[2]) * cos_(x[1]) - cos_(x[1] * x[2]) * sin_(x[1]);
    H[2][2] = pow(x[1], 2) * sin_(x[1] * x[2]) * sin_(x[1]);

    // std::cout << '\n';

    for (int i = 1; i <= n; i++) {
        som = 0;

        for (int j = 1; j <= n; j++) {
            if (i != j) {
                if (abs(H[i][j].lower()) > abs(H[i][j].upper())) { max = abs(H[i][j].lower()); }
                else { max = abs(H[i][j].upper()); }
                som = som + max * (dis[j] / dis[i]);
            }
        }
        rest = H[i][i].lower() - som;

        if (rest < 0) { k[i] = -rest; }
        else { k[i] = 0; }
    }

    for (int i = 1; i <= n; i++) {
        //if (k[i] > 100) { k[i] = 100; }
      // std::cout << "alpha" << i << "= " << k[i] << "   ";
    }

    //std::cout << '\n';
}
void Gershgorink(I x[5], int n) {
    double max, rest, som, dis[5];
    I H[5][5];
    for (int i = 1; i <= n; i++) {
        dis[i] = x[i].upper() - x[i].lower();
        if (dis[i] == 0) { dis[i] = 0.000001; }
        ///std::cout << dis[i] << "  ";
         //dis[i] = 1;
    }
    // f1
    H[1][1] = sin_(x[1] * x[2]) * sin_(x[1]) - I(2) * x[2] * cos_(x[1] * x[2]) * cos_(x[1]) + pow(x[2], 2) * sin_(x[1] * x[2]) * sin_(x[1]);
    H[1][2] = x[1] * x[2] * sin_(x[1] * x[2]) * sin_(x[1]) - x[1] * cos_(x[1] * x[2]) * cos_(x[1]) - cos_(x[1] * x[2]) * sin_(x[1]);
    H[2][1] = x[1] * x[2] * sin_(x[1] * x[2]) * sin_(x[1]) - x[1] * cos_(x[1] * x[2]) * cos_(x[1]) - cos_(x[1] * x[2]) * sin_(x[1]);
    H[2][2] = pow(x[1], 2) * sin_(x[1] * x[2]) * sin_(x[1]);


    // std::cout << '\n';

    for (int i = 1; i <= n; i++) {
        som = 0;

        for (int j = 1; j <= n; j++) {
            if (i != j) {
                if (abs(H[i][j].lower()) > abs(H[i][j].upper())) { max = abs(H[i][j].lower()); }
                else { max = abs(H[i][j].upper()); }
                som = som + max * (dis[j] / dis[i]);
            }
        }
        rest = H[i][i].lower() - som;

        if (rest < 0) { k_v[i] = -rest; }
        else { k_v[i] = 0; }
    }

    for (int i = 1; i <= n; i++) {
        //if (k_v[i] > 100) { k_v[i] = 100; }
       // std::cout << "alpha_v" << i << "= " << k_v[i] << "   ";
    }

    //std::cout << '\n';
}
void Kqmax(I x[5]) {
    I H[5][5];
    I kq1[3];

    //f1
    H[1][1] = sin_(x[1] * x[2]) * sin_(x[1]) - I(2) * x[2] * cos_(x[1] * x[2]) * cos_(x[1]) + pow(x[2], 2) * sin_(x[1] * x[2]) * sin_(x[1]);
    H[1][2] = x[1] * x[2] * sin_(x[1] * x[2]) * sin_(x[1]) - x[1] * cos_(x[1] * x[2]) * cos_(x[1]) - cos_(x[1] * x[2]) * sin_(x[1]);
    H[2][1] = x[1] * x[2] * sin_(x[1] * x[2]) * sin_(x[1]) - x[1] * cos_(x[1] * x[2]) * cos_(x[1]) - cos_(x[1] * x[2]) * sin_(x[1]);
    H[2][2] = pow(x[1], 2) * sin_(x[1] * x[2]) * sin_(x[1]);


    kq1[1] = abs(H[1][1]) + abs(H[1][2]);
    kq1[2] = abs(H[2][1]) + abs(H[2][2]);
    /*
    for (int i = 1; i <= 2; i++) {
        using namespace io_std;  std::cout << kq1[i];
        std::cout << '\n';
    }
    */
    if (kq1[1].upper() > kq1[2].upper()) {
        K1 = kq1[1].upper();
    }
    else { K1 = kq1[2].upper(); }
    //std::cout << "Kq= " << K1 << "\n\n";

}

double fx(double x[5]) {
    double f;
     f = -sin(x[1]) * sin(x[1] * x[2]);
    //f2 f = -sin((2 * x[1]) + x[2]) / (sin(x[2]) + 2);
    //f3 f = sin(x[1] + x[2]) + pow(x[1] - x[2], 2) - (1.5 * x[1]) + (2.5 * x[2]) + 1;
    //f4 f = -sin((x[1] - 1) * (x[1] - 2) * (x[2] + 1));
    //f5 f = pow(x[1] - 2, 2) + pow((x[2] - 1), 2) + 0.04 / (1 - (pow(x[1], 2)) / 4 - pow(x[2], 2)) + (pow((x[1] - 2 * x[2] + 1), 2) / 0.2);
    //f6 f = (pow((x[2] - 5 * (pow(x[1], 2)) / (4 * pow((3.14), 2)) + (5 * x[1]) / (3.14) - 6), 2)) + 10 * (1 - 1 / (8 * (3.14))) * cos(x[1]) + 10;
    //f7 f = 100 * pow((x[2] - pow(x[1], 2)), 2) + pow((x[1] - 1), 2);
    //f8 f = 0.1 * (12 + pow(x[1], 2) + (1 + pow(x[2], 2)) / (pow(x[1], 2))) + ((pow(x[1], 2)) * (pow(x[2], 2)) + 100) / pow((x[1] * x[2]), 4);
    //f9 f = 0.5 * (pow(x[1], 2) + pow(x[2], 2)) - cos(10 * log(2 * x[1])) * cos(10 * log(3 * x[2])) + 1;
    //f10 f = (1 + pow((x[1] + x[2] + 1), 2) * (19 - 14 * x[1] + 3 * pow(x[1], 2) - 14 * x[2] + 6 * x[1] * x[2] + 3 * pow(x[2], 2))) * (30 + pow((2 * x[1] - 3 * x[2]), 2) * (18 - 32 * x[1] + 12 * pow(x[1], 2) + 48 * x[2] - 36 * x[1] * x[2] + 27 * pow(x[2], 2)));
    //f11 f = pow(x[1], 4) + x[2] - pow((x[1] + pow(x[2], 2)), 2);
    //f12 f= pow(1 + x[1] -exp(x[2]), 2);
    //f13 f = pow((2 * x[1] + x[2] - 3), 2) + pow((x[1] * x[2] - 1), 2);
    //f14 f = 4 * (pow(x[1], 2)) - 2.1 * (pow(x[1], 4)) + (1 / 3) * (pow(x[1], 6)) + x[1] * x[2] - 4 * (pow(x[2], 2)) + 4 * (pow(x[2], 4));
    //f15 f = pow((x[1] - 1), 2) + pow((x[2] - 1), 2) + pow((x[1] + 2 * x[2] - 3), 2) + pow((x[1] + 2 * x[2] - 3), 4);
    //f16 f= (pow(x[1] , 2) - 0.1 * cos(5 * pii * x[1])) + (pow(x[2] , 2) - 0.1 * cos(5 * pii * x[2]));
    return f;
}
double fx0(double x1, double x2) {
    double f;
     f = -sin(x1) * sin(x1 * x2);
    //f2 f = -sin((2 * x1) + x2) / (sin(x2) + 2);
    //f3 f = sin(x1 + x2) + pow(x1 - x2, 2) - (1.5 * x1) + (2.5 * x2) + 1;
    //f4 f = -sin((x1 - 1) * (x1 - 2) * (x2 + 1));
    //f5 f = pow(x1 - 2, 2) + pow((x2 - 1), 2) + 0.04 / (1 - (pow(x1, 2)) / 4 - pow(x2, 2)) + (pow((x1 - 2 * x2 + 1), 2) / 0.2);
    //f6 f = (pow((x2 - 5 * (pow(x1, 2)) / (4 * pow((3.14), 2)) + (5 * x1) / (3.14) - 6), 2)) + 10 * (1 - 1 / (8 * (3.14))) * cos(x1) + 10;
    //f7 f = 100 * pow((x2 - pow(x1, 2)), 2) + pow((x1 - 1), 2);
    //f8 f = 0.1 * (12 + pow(x1, 2) + (1 + pow(x2, 2)) / (pow(x1, 2))) + ((pow(x1, 2)) * (pow(x2, 2)) + 100) / pow((x1 * x2), 4);
    //f9 f = 0.5 * (pow(x1, 2) + pow(x2, 2)) - cos(10 * log(2 * x1)) * cos(10 * log(3 * x2)) + 1;
    //f10 f = (1 + pow((x1 + x2 + 1), 2) * (19 - 14 * x1 + 3 * pow(x1, 2) - 14 * x2 + 6 * x1 * x2 + 3 * pow(x2, 2))) * (30 + pow((2 * x1 - 3 * x2), 2) * (18 - 32 * x1 + 12 * pow(x1, 2) + 48 * x2 - 36 * x1 * x2 + 27 * pow(x2, 2)));
    //f11 f = pow(x1, 4) + x2 - pow((x1 + pow(x2, 2)), 2);
    //f12 f= pow(1 + x1 -exp(x2), 2);
    //f13 f = pow((2 * x1 + x2 - 3), 2) + pow((x1 * x2 - 1), 2);
    //f14 f = 4 * (pow(x1, 2)) - 2.1 * (pow(x1, 4)) + (1 / 3) * (pow(x1, 6)) + x1 * x2 - 4 * (pow(x2, 2)) + 4 * (pow(x2, 4));
    //f15 f = pow((x1 - 1), 2) + pow((x2 - 1), 2) + pow((x1 + 2 * x2 - 3), 2) + pow((x1 + 2 * x2 - 3), 4);
    //f16 f= (pow(x1 , 2) - 0.1 * cos(5 * pii * x1)) + (pow(x2 , 2) - 0.1 * cos(5 * pii * x2));
    return f;
}

void gradlbp(double x[3], I x1[5], double seg[5]) {


    /*
     df1
     grad[0] = -sin(x[0] * x[1]) * cos(x[0]) - x[1] * cos(x[0] * x[1]) * sin(x[0]);
     grad[1] = -x[0] * cos(x[0] * x[1]) * sin(x[0]);
     df2
     grad[0] = -(2 * cos(2 * x[0] + x[1])) / (sin(x[1]) + 2);
     grad[1] = (sin(2 * x[0] + x[1]) * cos(x[1])) / pow(sin(x[1]) + 2, 2) - cos(2 * x[0] + x[1]) / (sin(x[1]) + 2);
     df3
     grad[0] = 2 * x[0] - 2 * x[1] + cos(x[0] + x[1]) - (1.5) ;
     grad[1] = 2 * x[1] - 2 * x[0] + cos(x[0] + x[1]) + (2.5) ;
     df4
     grad[0] = -cos((x[0] - 1) * (x[0] - 2) * (x[1] + 1)) * ((x[0] - 1) * (x[1] + 1) + (x[0] - 2) * (x[1] + 1));
     grad[1] = -cos((x[0] - 1) * (x[0] - 2) * (x[1] + 1)) * (x[0] - 1) * (x[0] - 2);
     df5
     grad[0] = 12 * x[0] - 20 * x[1] + x[0] / (50 * pow((pow(x[0], 2) / 4 + pow(x[1], 2) - 1), 2)) + 6;
     grad[1] = 42 * x[1] - 20 * x[0] + (2 * x[1]) / (25 * pow((pow(x[0], 2) / 4 + pow(x[1], 2) - 1), 2)) - 22;
     df6
     grad[0] = -(3015 * sin(x[0])) / 314 - 2 * ((6250 * x[0]) / 24649 - 250 / 157) * ((250 * x[0]) / 157 + x[1] - (3125 * pow(x[0],2)) / 24649 - 6) ;
     grad[1] = (500 * x[0]) / 157 + 2 * x[1] - (6250 * pow(x[0], 2)) / 24649 - 12;
     df7
     grad[0] = 2 * x[0] - 400 * x[0] * (-pow(x[0], 2) + x[1]) - 2;
     grad[1] = -200 * pow(x[0], 2) + 200 * x[1];
     df8
     grad[0] = x[0] / 5 + 2 / (pow(x[0], 3) * pow(x[1], 2)) - (pow(x[1], 2) + 1) / (5 * pow(x[0], 3)) - (4 * (pow(x[0], 2) * pow(x[1], 2) + 100)) / (pow(x[0], 5) * pow(x[1], 4));
     grad[1] = 2 / (pow(x[0] , 2) * pow(x[1] , 3)) + x[1] / (5 * pow(x[0] , 2)) - (4 * (pow(x[0] , 2) * pow(x[1] , 2) + 100)) / (pow(x[0] , 4) * pow(x[1] , 5));
     df9
     grad[0] = x[0] + (10 * cos(10 * log(3 * x[1])) * sin(10 * log(2 * x[0]))) / x[0];
     grad[1] = x[1] + (10 * cos(10 * log(2 * x[0])) * sin(10 * log(3 * x[1]))) / x[1];
     df10
     grad[0] = ((6 * x[0] + 6 * x[1] - 14) * pow((x[0] + x[1] + 1), 2) + (2 * x[0] + 2 * x[1] + 2) * (3 * pow(x[0], 2) + 6 * x[0] * x[1] - 14 * x[0] + 3 * pow(x[1], 2) - 14 * x[1] + 19)) * (pow((2 * x[0] - 3 * x[1]), 2) * (12 * pow(x[0], 2) - 36 * x[0] * x[1] - 32 * x[0] + 27 * pow(x[1], 2) + 48 * x[1] + 18) + 30) + (pow((x[0] + x[1] + 1), 2) * (3 * pow(x[0], 2) + 6 * x[0] * x[1] - 14 * x[0] + 3 * pow(x[1], 2) - 14 * x[1] + 19) + 1) * ((8 * x[0] - 12 * x[1]) * (12 * pow(x[0], 2) - 36 * x[0] * x[1] - 32 * x[0] + 27 * pow(x[1], 2) + 48 * x[1] + 18) - pow((2 * x[0] - 3 * x[1]), 2) * (36 * x[1] - 24 * x[0] + 32));
     grad[1] = ((6 * x[0] + 6 * x[1] - 14) * pow((x[0] + x[1] + 1), 2) + (2 * x[0] + 2 * x[1] + 2) * (3 * pow(x[0], 2) + 6 * x[0] * x[1] - 14 * x[0] + 3 * pow(x[1], 2) - 14 * x[1] + 19)) * (pow((2 * x[0] - 3 * x[1]), 2) * (12 * pow(x[0], 2) - 36 * x[0] * x[1] - 32 * x[0] + 27 * pow(x[1], 2) + 48 * x[1] + 18) + 30) - (pow((x[0] + x[1] + 1), 2) * (3 * pow(x[0], 2) + 6 * x[0] * x[1] - 14 * x[0] + 3 * pow(x[1], 2) - 14 * x[1] + 19) + 1) * ((12 * x[0] - 18 * x[1]) * (12 * pow(x[0], 2) - 36 * x[0] * x[1] - 32 * x[0] + 27 * pow(x[1], 2) + 48 * x[1] + 18) - pow((2 * x[0] - 3 * x[1]), 2) * (54 * x[1] - 36 * x[0] + 48));
         df11
         grad[0] = 4 * pow(x[0], 3) - 2 * x[0] - 2 * pow(x[1], 2);
         grad[1] = 1 - 4 * x[1] * (pow(x[1], 2) + x[0]);
         df12
         grad[0] = 2 * x[0] - 2 * exp(x[1]) + 2;
         grad[1] =  -2 * exp(x[1]) * (x[0] - exp(x[1]) + 1);
         df13
         grad[0] = 8 * x[0] + 4 * x[1] + 2 * x[1] * (x[0] * x[1] - 1) - 12;
         grad[1] = 4 * x[0] + 2 * x[1] + 2 * x[0] * (x[0] * x[1] - 1) - 6;
         df14
         grad[0] = 2 * pow(x[0],5) - (42 * pow(x[0],3)) / 5 + 8 * x[0] + x[1];
         grad[1] = 16 * pow(x[1], 3) - 8 * x[1] + x[0];
         df15
         grad[0] = 4 * x[0] + 4 * x[1] + 4 * pow((x[0] + 2 * x[1] - 3), 3);
         grad[1] = 4 * x[0] + 10 * x[1] + 8 * pow((x[0] + 2 * x[1] - 3), 3) - 14;
         df16
         grad[0] = 2 * x[0] + (pii * sin(5 * pii * x[0])) / 2;
         grad[1] = 2 * x[1] + (pii * sin(5 * pii * x[1])) / 2;
        
     */
}
double dLB(I x1[5], double k_alpha[5], int n) {
    double distance = 0;
    for (int i = 1; i <= n; i++) {
        distance = distance + ((k_alpha[i] / 8) * pow(x1[i].upper() - x1[i].lower(), 2));
    }
    return distance;
}

double LBalpha(double x[5], I x1[5], double k[5]) {
    double lb;
    lb = fx(x) - (0.5) * ((k[1] * ((x1[1].upper() - x[1]) * (x[1] - x1[1].lower()))) + (k[2] * ((x1[2].upper() - x[2]) * (x[2] - x1[2].lower()))));

    return lb;
}
double LB_gamma(double x[5], I x1[5], double gam[5]) {
    double lb;
    lb = fx(x) - (1 - exp(gam[1] * (x[1] - x1[1].lower())) * (1 - exp(gam[1] * (x1[1].upper() - x[1])))) - (1 - exp(gam[2] * (x[2] - x1[2].lower())) * (1 - exp(gam[2] * (x1[2].upper() - x[2]))));

    return lb;
}
double w0(I x, double x1) {
    double w;

    w = (x.upper() - x1) / (x.upper() - x.lower());

    return w;

}

double w1(I x, double x1) {
    double w;

    w = (x1 - x.lower()) / (x.upper() - x.lower());

    return w;

}
double LB_q(double x1[3], I x[5], double k1) {
    double q, s1[3], s2[3], s3[3], s4[4];

    s1[1] = x[1].lower(); s1[2] = x[2].lower();
    s2[1] = x[1].upper(); s2[2] = x[2].lower();
    s3[1] = x[1].lower(); s3[2] = x[2].upper();
    s4[1] = x[1].upper(); s4[2] = x[2].upper();

    q = fx(s1) * w0(x[1], x1[1]) * w0(x[2], x1[2]) + fx(s2) * w1(x[1], x1[1]) * w0(x[2], x1[2]) + fx(s3) * w0(x[1], x1[1]) * w1(x[2], x1[2]) + fx(s4) * w1(x[1], x1[1]) * w1(x[2], x1[2]) - (0.5)*(k1 * ((x1[1] - x[1].lower()) * (x[1].upper() - x1[1]) + (x1[2] - x[2].lower()) * (x[2].upper() - x1[2])));

    return q;
}
double LB_p(double x[5], I x1[5], double seg[5]) {
    double lb;
    lb = fx(x) - (seg[1] * (((log(((x[1] - x1[1].lower()) / (x1[1].upper() - x1[1].lower())) + 1) + log(((x1[1].upper() - x[1]) / (x1[1].upper() - x1[1].lower())) + 1)) / log(2)) - 1)) - (seg[2] * (((log(((x[2] - x1[2].lower()) / (x1[2].upper() - x1[2].lower())) + 1) + log(((x1[2].upper() - x[2]) / (x1[2].upper() - x1[2].lower())) + 1)) / log(2)) - 1));

    return lb;
}

double betaik(I xp[8][5], double alphak[8][5], int i, int k, int p) {
    double bet1, betk, sum1 = 0, sumk = 0;
    for (int j = 0; j <= p - 2; j++) {
        sum1 = sum1 - (alphak[j][i] * (xp[j][i].upper() - xp[j][i].lower()) + alphak[j + 1][i] * (xp[j + 1][i].upper() - xp[j + 1][i].lower())) * (xp[j][i].upper() - xp[p - 1][i].upper());
    }
    bet1 = sum1 / (xp[p - 1][i].upper() - xp[0][i].lower());
    if (k != 0) {
        for (int j = 0; j <= k - 1; j++) {
            sumk = sumk + (-alphak[j][i] * (xp[j][i].upper() - xp[j][i].lower()) - alphak[j + 1][i] * (xp[j + 1][i].upper() - xp[j + 1][i].lower()));
        }
        betk = sumk + bet1;
    }
    // std::cout << "beta1= " << bet1 << '\n';
    if (k == 0) { return bet1; }
    else {
        //     std::cout << "betk= " << betk << '\n';
        return betk;
    }
}

double gammaik(I xp[8][5], double alphak[8][5], int i, int k, int p) {
    double bet1, gammk, sum1 = 0, sumk = 0;
    for (int j = 0; j <= p - 2; j++) {
        sum1 = sum1 - (alphak[j][i] * (xp[j][i].upper() - xp[j][i].lower()) + alphak[j + 1][i] * (xp[j + 1][i].upper() - xp[j + 1][i].lower())) * (xp[j][i].upper() - xp[p - 1][i].upper());
    }
    bet1 = sum1 / (xp[p - 1][i].upper() - xp[0][i].lower());
    if (k != 0) {
        for (int j = 0; j <= k - 1; j++) {
            sumk = sumk - ((alphak[j][i] * (xp[j][i].upper() - xp[j][i].lower()) + alphak[j + 1][i] * (xp[j + 1][i].upper() - xp[j + 1][i].lower())) * xp[j][i].upper());
        }
    }
    gammk = -sumk - (bet1 * xp[0][i].lower());
    // std::cout << "beta1//= " << bet1 << '\n';
    // std::cout << "gammk= " << gammk << '\n';
    return gammk;
}

double LB_piecewise_q(double x[5], I x1[5], int p, int n) {
    double lb, sum = 0, pieces[5]; double alpha[8][5];
    int jk;
    I xp[8][5];
    for (int i = 1; i <= n; i++) { pieces[i] = (x1[i].upper() - x1[i].lower()) / p; }
    for (int j = 0; j <= p - 1; j++) {
        for (int i = 1; i <= n; i++) {
            xp[j][i].set(x1[i].lower() + j * pieces[i], x1[i].lower() + (j + 1) * pieces[i]);
        }
        Gershgorink(xp[j], n);
        for (int i = 1; i <= n; i++) { alpha[j][i] = k_v[i]; }
    }/*
for (int j = 0; j <= p - 1; j++) {
    for (int i = 1; i <= n; i++) {
        using namespace io_std;  std::cout << xp[j][i] << '\n';
    }
    std::cout << "\n\n";
}*/

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= p - 1; j++) {
            if (x[i] >= xp[j][i].lower() && x[i] <= xp[j][i].upper()) {
                jk = j;
                j = p + 1;
                // std::cout << "jk= "<<jk;
            }
        }
        sum = sum + (alpha[jk][i] * (x[i] - xp[jk][i].lower()) * (xp[jk][i].upper() - x[i]) + betaik(xp, alpha, i, jk, p) * x[i] + gammaik(xp, alpha, i, jk, p));
    }

    lb = fx(x) - (0.5) * sum;
    return lb;
}

double myfunc3(unsigned n, const double* x, double* grad, void* my_func_data)
{
    double sum = 0, alpha[8][5], pieces[5];
    int jk[5];
    I xp[8][5];
    for (int i = 1; i <= n; i++) { pieces[i] = (xg[i].upper() - xg[i].lower()) / Ni; }
    for (int j = 0; j <= Ni - 1; j++) {
        for (int i = 1; i <= n; i++) {
            xp[j][i].set(xg[i].lower() + j * pieces[i], xg[i].lower() + (j + 1) * pieces[i]);
        }
        Gershgorink(xp[j], n);
        for (int i = 1; i <= n; i++) { alpha[j][i] = k_v[i]; }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= Ni - 1; j++) {
            if (x[i] >= xp[j][i + 1].lower() && x[i] <= xp[j][i + 1].upper()) {
                jk[i] = j;
                j = Ni + 1;
            }
        }
    }
    if (grad) {
        grad[0] = -sin(x[0] * x[1]) * cos(x[0]) - x[1] * cos(x[0] * x[1]) * sin(x[0]) - alpha[jk[0]][1] * (-2 * x[0] + xp[jk[0]][1].upper() + xp[jk[0]][1].lower()) - betaik(xp, alpha, 1, jk[0], Ni);
        grad[1] = -x[0] * cos(x[0] * x[1]) * sin(x[0]) - alpha[jk[1]][2] * (-2 * x[1] + xp[jk[1]][2].upper() + xp[jk[1]][2].lower()) - betaik(xp, alpha, 2, jk[1], Ni);
    }
    for (int i = 1; i <= n; i++) {
        sum = sum + (alpha[jk[i - 1]][i] * (x[i - 1] - xp[jk[i - 1]][i].lower()) * (xp[jk[i - 1]][i].upper() - x[i - 1]) + betaik(xp, alpha, i, jk[i - 1], Ni) * x[i - 1] + gammaik(xp, alpha, i, jk[i - 1], Ni));
    }
    return -sin(x[0]) * sin(x[0] * x[1]) - (0.5) * sum;
}
double myfunc4(unsigned n, const double* x, double* grad, void* my_func_data)
{
    if (grad) {
        grad[0] = -sin(x[0] * x[1]) * cos(x[0]) - x[1] * cos(x[0] * x[1]) * sin(x[0]) - (sigma[1] * ((1 / ((xg[1].upper() - xg[1].lower()) * (((x[0] - xg[1].lower()) / (xg[1].upper() - xg[1].lower())) + 1)) - 1 / ((xg[1].upper() - xg[1].lower()) * (((xg[1].upper() - x[0]) / (xg[1].upper() - xg[1].lower())) + 1))) / log(2)));
        grad[1] = -x[0] * cos(x[0] * x[1]) * sin(x[0]) - (sigma[2] * ((1 / ((xg[2].upper() - xg[2].lower()) * (((x[1] - xg[2].lower()) / (xg[2].upper() - xg[2].lower())) + 1)) - 1 / ((xg[2].upper() - xg[2].lower()) * (((xg[2].upper() - x[1]) / (xg[2].upper() - xg[2].lower())) + 1))) / log(2)));
    }
    return -sin(x[0]) * sin(x[0] * x[1]) - (sigma[1] * (((log(((x[0] - xg[1].lower()) / (xg[1].upper() - xg[1].lower())) + 1) + log(((xg[1].upper() - x[0]) / (xg[1].upper() - xg[1].lower())) + 1)) / log(2)) - 1)) - (sigma[2] * (((log(((x[1] - xg[2].lower()) / (xg[2].upper() - xg[2].lower())) + 1) + log(((xg[2].upper() - x[1]) / (xg[2].upper() - xg[2].lower())) + 1)) / log(2)) - 1));
}
// interval xg
double myfunc5(unsigned n, const double* x, double* grad, void* my_func_data)
{
    if (grad) {
        grad[0] = fx0(xg[1].lower(), xg[2].lower()) * (-1/(xg[1].upper()-xg[1].lower())) * w0(xg[2], x[1]) + fx0(xg[1].upper(), xg[2].lower()) * (1 / (xg[1].upper() - xg[1].lower())) * w0(xg[2], x[1]) + fx0(xg[1].lower(), xg[2].upper()) * (-1 / (xg[1].upper() - xg[1].lower())) * w1(xg[2], x[1]) + fx0(xg[1].upper(), xg[2].upper()) * (1 / (xg[1].upper() - xg[1].lower())) * w1(xg[2], x[1]) + K1 * (-xg[1].upper() - xg[1].lower() + 2 * x[0]);
        grad[1] = fx0(xg[1].lower(), xg[2].lower()) * w0(xg[1], x[0]) * (-1 / (xg[2].upper() - xg[2].lower())) + fx0(xg[1].upper(), xg[2].lower()) * w1(xg[1], x[0]) * (-1 / (xg[2].upper() - xg[2].lower())) + fx0(xg[1].lower(), xg[2].upper()) * w0(xg[1], x[0]) * (1 / (xg[2].upper() - xg[2].lower())) + fx0(xg[1].upper(), xg[2].upper()) * w1(xg[1], x[0]) * (1 / (xg[2].upper() - xg[2].lower())) + K1 * (-xg[2].upper() - xg[2].lower() + 2 * x[1]);

    }
    return  fx0(xg[1].lower(),xg[2].lower()) * w0(xg[1], x[0]) * w0(xg[2], x[1]) + fx0(xg[1].upper(), xg[2].lower()) * w1(xg[1], x[0]) * w0(xg[2], x[1]) + fx0(xg[1].lower(), xg[2].upper()) * w0(xg[1], x[0]) * w1(xg[2], x[1]) + fx0(xg[1].upper(), xg[2].upper()) * w1(xg[1], x[0]) * w1(xg[2], x[1]) - (0.5) * (K1 * ((x[0] - xg[1].lower()) * (xg[1].upper() - x[0]) + (x[1] - xg[2].lower()) * (xg[2].upper() - x[1])));
}
double myfunc6(unsigned n, const double* x, double* grad, void* my_func_data)
{
    if (grad) {
        grad[0] = -sin(x[0] * x[1]) * cos(x[0]) - x[1] * cos(x[0] * x[1]) * sin(x[0]) - (-gamma[1] * exp(gamma[1] * (x[0] - xg[1].lower())) + gamma[1] * exp(gamma[1] * (xg[1].upper() - x[0])));
        grad[1] = -x[0] * cos(x[0] * x[1]) * sin(x[0]) - (-gamma[2] * exp(gamma[2] * (x[1] - xg[2].lower())) + gamma[2] * exp(gamma[2] * (xg[2].upper() - x[1])));
    }
    return -sin(x[0]) * sin(x[0] * x[1]) - (1 - exp(gamma[1] * (x[0] - xg[1].lower())) * (1 - exp(gamma[1] * (xg[1].upper() - x[0])))) - (1 - exp(gamma[2] * (x[1] - xg[2].lower())) * (1 - exp(gamma[2] * (xg[2].upper() - x[1]))));
}
// interval xg
double myfunc1(unsigned n, const double* x, double* grad, void* my_func_data)
{
    if (grad) {
        grad[0] = -sin(x[0] * x[1]) * cos(x[0]) - x[1] * cos(x[0] * x[1]) * sin(x[0]) + k[1] * (-xg[1].upper() - xg[1].lower() + 2 * x[0]);
        grad[1] = -x[0] * cos(x[0] * x[1]) * sin(x[0]) + k[2] * (-xg[2].upper() - xg[2].lower() + 2 * x[1]);

    }
    return -sin(x[0]) * sin(x[0] * x[1]) - (0.5) * ((k[1] * ((xg[1].upper() - x[0]) * (x[0] - xg[1].lower()))) + (k[2] * ((xg[2].upper() - x[1]) * (x[1] - xg[2].lower()))));
}

//interval tg
double myfunc2(unsigned n, const double* x, double* grad, void* my_func_data)
{
    if (grad) {
        grad[0] = -sin(x[0] * x[1]) * cos(x[0]) - x[1] * cos(x[0] * x[1]) * sin(x[0]);
        grad[1] = -x[0] * cos(x[0] * x[1]) * sin(x[0]);
    }
    return -sin(x[0]) * sin(x[0] * x[1]);
}


void local_min(double si[5], int numb) {
    if (numb == 1) {
        double lb[2] = { xg[1].lower(), xg[2].lower() }; /* lower bounds */
        double ub[2] = { xg[1].upper(), xg[2].upper() }; /* upper bounds */
        nlopt_opt opt;

        opt = nlopt_create(NLOPT_LD_MMA, 2); /* algorithm and dimensionality */
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);
        nlopt_set_min_objective(opt, myfunc1, NULL);

        nlopt_set_xtol_rel(opt, 1e-4);

        double xi[2] = { si[1], si[2] };  /* `*`some` `initial` `guess`*` */
        double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
        if (nlopt_optimize(opt, xi, &minf) < 0) {
            printf("nlopt failed!\n");
        }
        else {
            printf("found minimum at LB_alpha(%g,%g) = %0.10g\n", xi[0], xi[1], minf);
        }
        sbar[1] = xi[0];
        sbar[2] = xi[1];

        /*
        if (sbar[1] == xg[1].upper()) { sbar[1] = sbar[1] - 0.00000001; }
        if (sbar[1] == xg[1].lower()) { sbar[1] = sbar[1] + 0.00000001; }
        if (sbar[2] == xg[2].upper()) { sbar[2] = sbar[2] - 0.00000001; }
        if (sbar[2] == xg[2].lower()) { sbar[2] = sbar[2] + 0.00000001; }
        */
    }
    else if (numb == 2) {
        double lb[2] = { xg[1].lower(), xg[2].lower() }; /* lower bounds */
        double ub[2] = { xg[1].upper(), xg[2].upper() }; /* upper bounds */
        nlopt_opt opt;

        opt = nlopt_create(NLOPT_LD_MMA, 2); /* algorithm and dimensionality */
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);
        nlopt_set_min_objective(opt, myfunc2, NULL);

        nlopt_set_xtol_rel(opt, 1e-4);

        double xi[2] = { si[1], si[2] };  /* `*`some` `initial` `guess`*` */
        double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
        if (nlopt_optimize(opt, xi, &minf) < 0) {
            printf("nlopt failed!\n");
        }
        else {
            printf("found minimum at f(%g,%g) = %0.10g\n", xi[0], xi[1], minf);
        }
        sbar[1] = xi[0];
        sbar[2] = xi[1];

        /*
        if (sbar[1] == xg[1].upper()) { sbar[1] = sbar[1] - 0.00000001; }
        if (sbar[1] == xg[1].lower()) { sbar[1] = sbar[1] + 0.00000001; }
        if (sbar[2] == xg[2].upper()) { sbar[2] = sbar[2] - 0.00000001; }
        if (sbar[2] == xg[2].lower()) { sbar[2] = sbar[2] + 0.00000001; }
        */
    }
    else if (numb == 3) {
        double lb[2] = { xg[1].lower(), xg[2].lower() }; /* lower bounds */
        double ub[2] = { xg[1].upper(), xg[2].upper() }; /* upper bounds */
        nlopt_opt opt;

        opt = nlopt_create(NLOPT_LD_MMA, 2); /* algorithm and dimensionality */
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);
        nlopt_set_min_objective(opt, myfunc3, NULL);

        nlopt_set_xtol_rel(opt, 1e-4);

        double xi[2] = { si[1], si[2] };  /* `*`some` `initial` `guess`*` */
        double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
        if (nlopt_optimize(opt, xi, &minf) < 0) {
            printf("nlopt failed!\n");
        }
        else {
            printf("found minimum at LB_piecewise(%g,%g) = %0.10g\n", xi[0], xi[1], minf);
        }
        sbar[1] = xi[0];
        sbar[2] = xi[1];

        /*
        if (sbar[1] == xg[1].upper()) { sbar[1] = sbar[1] - 0.00000001; }
        if (sbar[1] == xg[1].lower()) { sbar[1] = sbar[1] + 0.00000001; }
        if (sbar[2] == xg[2].upper()) { sbar[2] = sbar[2] - 0.00000001; }
        if (sbar[2] == xg[2].lower()) { sbar[2] = sbar[2] + 0.00000001; }
        */
    }
    else if (numb == 4) {
        double lb[2] = { xg[1].lower(), xg[2].lower() }; /* lower bounds */
        double ub[2] = { xg[1].upper(), xg[2].upper() }; /* upper bounds */
        nlopt_opt opt;

        opt = nlopt_create(NLOPT_LD_MMA, 2); /* algorithm and dimensionality */
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);
        nlopt_set_min_objective(opt, myfunc4, NULL);

        nlopt_set_xtol_rel(opt, 1e-4);

        double xi[2] = { si[1], si[2] };  /* `*`some` `initial` `guess`*` */
        double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
        if (nlopt_optimize(opt, xi, &minf) < 0) {
            printf("nlopt failed!\n");
        }
        else {
            printf("found minimum at LB_proposed(%g,%g) = %0.10g\n", xi[0], xi[1], minf);
        }
        sbar[1] = xi[0];
        sbar[2] = xi[1];

        /*
        if (sbar[1] == xg[1].upper()) { sbar[1] = sbar[1] - 0.00000001; }
        if (sbar[1] == xg[1].lower()) { sbar[1] = sbar[1] + 0.00000001; }
        if (sbar[2] == xg[2].upper()) { sbar[2] = sbar[2] - 0.00000001; }
        if (sbar[2] == xg[2].lower()) { sbar[2] = sbar[2] + 0.00000001; }
        */
    }
    else if (numb == 5) {
        double lb[2] = { xg[1].lower(), xg[2].lower() }; /* lower bounds */
        double ub[2] = { xg[1].upper(), xg[2].upper() }; /* upper bounds */
        nlopt_opt opt;

        opt = nlopt_create(NLOPT_LD_MMA, 2); /* algorithm and dimensionality */
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);
        nlopt_set_min_objective(opt, myfunc5, NULL);

        nlopt_set_xtol_rel(opt, 1e-4);

        double xi[2] = { si[1], si[2] };  /* `*`some` `initial` `guess`*` */
        double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
        if (nlopt_optimize(opt, xi, &minf) < 0) {
            printf("nlopt failed!\n");
        }
        else {
            printf("found minimum at LB_q(%g,%g) = %0.10g\n", xi[0], xi[1], minf);
        }
        sbar[1] = xi[0];
        sbar[2] = xi[1];

        /*
        if (sbar[1] == xg[1].upper()) { sbar[1] = sbar[1] - 0.00000001; }
        if (sbar[1] == xg[1].lower()) { sbar[1] = sbar[1] + 0.00000001; }
        if (sbar[2] == xg[2].upper()) { sbar[2] = sbar[2] - 0.00000001; }
        if (sbar[2] == xg[2].lower()) { sbar[2] = sbar[2] + 0.00000001; }
        */
        }
    else if (numb == 6) {
        double lb[2] = { xg[1].lower(), xg[2].lower() }; /* lower bounds */
        double ub[2] = { xg[1].upper(), xg[2].upper() }; /* upper bounds */
        nlopt_opt opt;

        opt = nlopt_create(NLOPT_LD_MMA, 2); /* algorithm and dimensionality */
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);
        nlopt_set_min_objective(opt, myfunc6, NULL);

        nlopt_set_xtol_rel(opt, 1e-4);

        double xi[2] = { si[1], si[2] };  /* `*`some` `initial` `guess`*` */
        double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
        if (nlopt_optimize(opt, xi, &minf) < 0) {
            printf("nlopt failed!\n");
        }
        else {
            printf("found minimum at LB_gamma(%g,%g) = %0.10g\n", xi[0], xi[1], minf);
        }
        sbar[1] = xi[0];
        sbar[2] = xi[1];

        /*
        if (sbar[1] == xg[1].upper()) { sbar[1] = sbar[1] - 0.00000001; }
        if (sbar[1] == xg[1].lower()) { sbar[1] = sbar[1] + 0.00000001; }
        if (sbar[2] == xg[2].upper()) { sbar[2] = sbar[2] - 0.00000001; }
        if (sbar[2] == xg[2].lower()) { sbar[2] = sbar[2] + 0.00000001; }
        */
        }

}
void convexification_gamma(I x1[5], int n) {
    double  gammak[5], alphak[5], mu = 1.01;
    int J = 1, Jmax = pow(2, n) + 1, kcpt = 1, Acpt, Afull, cnvx = 0;
    I x1j[5], x1last[5], A1[500][5], equation[5];
    convex = 1;
    for (int i = 1; i <= n; i++) {
        x1j[i] = x1[i];
        A1[1][i] = x1j[i];
    }
    Acpt = 1; Afull = 1;
    Gershgorin(x1, n);
    for (int i = 1; i <= n; i++) {
        // sigmalk[i] = k[i] / (5 / (4 * pow(x1[i].upper() - x1[i].lower(), 2)) / log(2));
        // alphak[i] = (8 * sigmalk[i] / pow(x1[i].upper() - x1[i].lower(), 2)) * ((2 * log(1.5) / log(2)) - 1);
        K1 = k[i];
        L = x1[i].lower(); U = x1[i].upper();
        gammak[i] = calc_sigma();
        std::cout << "gamma "<<i<<"= " << gammak[1] << '\n';
        alphak[i] = (4 * pow((1 - exp(0.5 * gammak[i] * (x1[i].upper() - x1[i].lower()))), 2)) / pow((x1[i].upper() - x1[i].lower()), 2);
    }
   // std::cout << "dLB =" << dLB(x1, k, n) << '\n';
   // std::cout << "dLBc =" << dLB(x1,alphak,n) << '\n';
    while (dLB(x1, k, n) > dLB(x1, alphak, n) && cnvx == 0) {
        do {
            for (int i = 1; i <= n; i++) {
                x1last[i] = A1[Acpt][i];
            }
            //using namespace io_std;  std::cout << x1last << '\n';
            J = J - 1;
            Gershgorink(x1last, n);
            for (int i = 1; i <= n; i++) {
                //equation[i] = -I(k_v[i]) + I(alphak[i]);
                //equation[i] = -I(k_v[i]) + ((I(sigmalk[i]) / (I(log(2) * pow(x1last[i].upper() - x1last[i].lower(), 2)))) * ((I(1) / pow(((x1last[i] - I(x1last[i].lower())) / (x1last[i].upper() - x1last[i].lower())) + I(1), 2)) + (I(1) / pow(((I(x1last[i].upper()) - x1last[i]) / (x1last[i].upper() - x1last[i].lower())) + I(1), 2))));
                equation[i] = -I(k_v[i]) + I(pow(gammak[i], 2)) * exp_(I(gammak[i]) * (x1last[i] - I(x1last[i].lower()))) + I(pow(gammak[i], 2)) * exp_(I(gammak[i]) * (I(x1last[i].upper()) - x1last[i]));
               // std::cout << "equation " << i << " =" << equation[i].lower() << '\n';
                if (equation[i].lower() < 0) {
                    Afull++;
                    for (int b = 1; b <= n; b++) {
                        if (i == b) {
                            A1[Afull][i].set(x1last[i].lower(), (x1last[i].lower() + x1last[i].upper()) / 2);
                        }
                        else {
                            A1[Afull][b] = x1last[b];
                        }
                    }
                    Afull++;
                    for (int b = 1; b <= n; b++) {
                        if (i == b) {
                            A1[Afull][i].set((x1last[i].lower() + x1last[i].upper()) / 2, x1last[i].upper());
                        }
                        else {
                            A1[Afull][b] = x1last[b];
                        }
                    }
                    J = J + 2;
                }
            }
            Acpt++;
        } while (Afull > Acpt - 1 && J < Jmax);
        if (Afull == Acpt - 1) { cnvx = 1; }
        else {
            for (int i = 1; i <= n; i++) {
                gammak[i] = gammak[i] * mu; kcpt++; //J = 1;
                //std::cout << "sigmak "<<i<<"= " << sigmalk[i] << '\n';
                alphak[i] = (4 * pow((1 - exp(0.5 * gammak[i] * (x1[i].upper() - x1[i].lower()))), 2)) / pow((x1[i].upper() - x1[i].lower()), 2);
                // std::cout << "alphak " << i << "= " << alphak[i] << '\n';
            }
        }
        /*
        for (int f = 1; f <= Afull; f++) {
            for (int i = 1; i <= n; i++) {
                using namespace io_std;  std::cout << A1[f][i] << "    ";
            }std::cout << "\n";
        }std::cout << "\n\n";
        //std::cout << "dLB =" << dLB(x1, k, n) << '\n';
        //std::cout << "dLBc =" << dLB(x1, alphak, n) << '\n';*/
    }
    if (cnvx == 1) {
        std::cout << "/////////////////////// gamma underestimator ////////////////////////// \n";
        for (int i = 1; i <= n; i++) {
            gamma[i] = gammak[i];
            k[i] = alphak[i];
        }
    }
    else { for (int i = 1; i <= n; i++) { gamma[i] = 0; } convex = 0; }
}

void convexification(I x1[5], int n) {
    double  sigmalk[5], alphak[5], mu = 1.01;
    int J = 1, Jmax = pow(2, n) + 1, kcpt = 1, Acpt, Afull, cnvx = 0;
    I x1j[5], x1last[5], A1[500][5], equation[5];
    convex = 1;
    for (int i = 1; i <= n; i++) {
        x1j[i] = x1[i];
        A1[1][i] = x1j[i];
    }
    Acpt = 1; Afull = 1;
    Gershgorin(x1, n);
    for (int i = 1; i <= n; i++) {
        sigmalk[i] = k[i] / (5 / (4 * pow(x1[i].upper() - x1[i].lower(), 2)) / log(2));
        alphak[i] = (8 * sigmalk[i] / pow(x1[i].upper() - x1[i].lower(), 2)) * ((2 * log(1.5) / log(2)) - 1);
    }
    //std::cout << "dLB =" << dLB(x1, k, n) << '\n';
    //std::cout << "dLBc =" << dLB(x1,alphak,n) << '\n';
    while (dLB(x1, k, n) > dLB(x1, alphak, n) && cnvx == 0) {
        do {
            for (int i = 1; i <= n; i++) {
                x1last[i] = A1[Acpt][i];
            }
            //using namespace io_std;  std::cout << x1last << '\n';
            J = J - 1;
            Gershgorink(x1last, n);
            for (int i = 1; i <= n; i++) {
                //equation[i] = -I(k_v[i]) + I(alphak[i]);
                equation[i] = -I(k_v[i]) + ((I(sigmalk[i]) / (I(log(2) * pow(x1last[i].upper() - x1last[i].lower(), 2)))) * ((I(1) / pow(((x1last[i] - I(x1last[i].lower())) / (x1last[i].upper() - x1last[i].lower())) + I(1), 2)) + (I(1) / pow(((I(x1last[i].upper()) - x1last[i]) / (x1last[i].upper() - x1last[i].lower())) + I(1), 2))));
                //std::cout << "equation " << i << " =" << equation[i].lower() << '\n';
                if (equation[i].lower() < 0) {
                    Afull++;
                    for (int b = 1; b <= n; b++) {
                        if (i == b) {
                            A1[Afull][i].set(x1last[i].lower(), (x1last[i].lower() + x1last[i].upper()) / 2);
                        }
                        else {
                            A1[Afull][b] = x1last[b];
                        }
                    }
                    Afull++;
                    for (int b = 1; b <= n; b++) {
                        if (i == b) {
                            A1[Afull][i].set((x1last[i].lower() + x1last[i].upper()) / 2, x1last[i].upper());
                        }
                        else {
                            A1[Afull][b] = x1last[b];
                        }
                    }
                    J = J + 2;
                }
            }
            Acpt++;
        } while (Afull > Acpt - 1 && J < Jmax);
        if (Afull == Acpt - 1) { cnvx = 1; }
        else {
            for (int i = 1; i <= n; i++) {
                sigmalk[i] = sigmalk[i] * mu; kcpt++; //J = 1;
                //std::cout << "sigmak "<<i<<"= " << sigmalk[i] << '\n';
                alphak[i] = (8 * sigmalk[i] / pow(x1[i].upper() - x1[i].lower(), 2)) * ((2 * log(1.5) / log(2)) - 1);
                // std::cout << "alphak " << i << "= " << alphak[i] << '\n';
            }
        }
        /*
        for (int f = 1; f <= Afull; f++) {
            for (int i = 1; i <= n; i++) {
                using namespace io_std;  std::cout << A1[f][i] << "    ";
            }std::cout << "\n";
        }std::cout << "\n\n";
        //std::cout << "dLB =" << dLB(x1, k, n) << '\n';
        //std::cout << "dLBc =" << dLB(x1, alphak, n) << '\n';*/
    }
    if (cnvx == 1) {
        std::cout << "/////////////////////// proposed underestimator ////////////////////////// \n";
        for (int i = 1; i <= n; i++) {
            sigma[i] = sigmalk[i];
            k[i] = alphak[i];
        }
    }
    else { for (int i = 1; i <= n; i++) { sigma[i] = 0; } convex = 0; }
}

void min_UB(I x[5], double UBD, int n) {
    double sop[5];

    for (int i = 1; i <= n; i++) {
        sop[i] = (x[i].lower() + x[i].upper()) / 2;
        xg[i] = x[i];
    }
    local_min(sop, 2);
    if (fx(sbar) < UBD) {
        UB = fx(sbar);
        for (int i = 1; i <= n; i++) {
            sopt[i] = sbar[i];
        }
    }
}

void interval_split(I x[5], int n) {
    double longest, indice = 1;
    longest = x[1].upper() - x[1].lower();
    for (int i = 2; i <= n; i++) {
        if (longest < (x[i].upper() - x[i].lower())) {
            longest = x[i].upper() - x[i].lower();
            indice = i;
        }
    }

    for (int i = 1; i <= n; i++) {
        if (i == indice) {
            Tk[i][1].set(x[i].lower(), (x[i].upper() + x[i].lower()) / 2);
            Tk[i][2].set((x[i].upper() + x[i].lower()) / 2, x[i].upper());
        }
        else {
            Tk[i][1] = x[i];
            Tk[i][2] = x[i];
        }
    }

}

int main()
{

    typedef interval<double,
        policies<rounded_math<double>,
        checking_base<double> > > J;

    int choice, repeat, rtimes = 1;
    double tab[1010], avg = 0, sd = 0;
    std::cout << "AlphaBB underestimator=1; Proposed underestimator=2; Piecewise underestimator=3; Quadratic underestimator=4; gamma underestimator=5 \n";
    std::cin >> choice;
    if (choice == 3) {
        std::cout << "The number Ni of the pieceswise underestimator is 2 or 4?\n";
        std::cin >> Ni;
    }
    timing = 0;

    J x[5];
    double s[5], s3[5], eps = 0.0000001, Tcpu = 0;
    int i, l, jend, it = 2, iteration=1;
    double a, ai[5][5], ks[5];
    J M[10000][5];
    x[1].set(0, 4);
    x[2].set(0, 4);

    T[1] = x[1];
    T[2] = x[2];
    
    // using namespace io_std;  std::cout << T[1] << '\n';
    // using namespace io_std;  std::cout << T[2] << '\n';

    for (int j = 1; j <= n; j++) {
        xg[j] = x[j];
        s3[j] = (xg[j].upper() + xg[j].lower()) / 2;
    }
    local_min(s3, 2);

    switch (choice) {
    case 1:
        Gershgorin(xg, n);
        local_min(s3, 1);
        break;
    case 2:
        Gershgorin(xg, n);
        convexification(xg, n);
        if (convex == 1) {
            local_min(s3, 4);
        }
        else {
            local_min(s3, 1);
        }
        break;
    case 4:
        Kqmax(xg);
        local_min(s3, 5);
        break;
    case 5:
        Gershgorin(xg, n);
        convexification_gamma(xg, n);
        if (convex == 1) {
            local_min(s3, 6);
        }
        else {
            local_min(s3, 1);
        }
        break;

    }

    UB = 10000000000000;
    for (jend = 1; jend <= it; jend++) {
        min_UB(T, UB, n);
        interval_split(T, n);
        // using namespace io_std;  std::cout << Tk[1][1] << '\n';
        // using namespace io_std;  std::cout << Tk[2][1] << '\n';
        // std::cout << '\n';
        // using namespace io_std;  std::cout << Tk[1][2] << '\n';
       //  using namespace io_std;  std::cout << Tk[2][2] << '\n';

        for (i = 1; i <= 2; i++) {
            for (int j = 1; j <= n; j++) {
                xg[j] = Tk[j][i];
                s3[j] = (xg[j].upper() + xg[j].lower()) / 2;
            }
            switch (choice) {
            case 1:
                Gershgorin(xg, n);
                local_min(s3, 1);
                if (LBalpha(sbar, xg, k) < UB) {
                    LBk[it][1] = LBalpha(sbar, xg, k);
                    LBk[it][2] = 0;
                    for (int rl = 1; rl <= n; rl++) {
                        xl[it][rl] = sbar[rl];
                        M[it][rl] = xg[rl];
                    }
                    it++;
                }
                break;
            case 2:
                Gershgorin(xg, n);
                convexification(xg, n);
                if (convex == 1){
                    local_min(s3, 4);
                if (LB_p(sbar, xg, sigma) < UB) {
                    LBk[it][1] = LB_p(sbar, xg, sigma);
                    LBk[it][2] = 0;
                    for (int rl = 1; rl <= n; rl++) {
                        xl[it][rl] = sbar[rl];
                        M[it][rl] = xg[rl];
                    }
                    it++;
                }
                }
                else {
                    local_min(s3, 1);
                    if (LBalpha(sbar, xg, k) < UB) {
                        LBk[it][1] = LBalpha(sbar, xg, k);
                        LBk[it][2] = 0;
                        for (int rl = 1; rl <= n; rl++) {
                            xl[it][rl] = sbar[rl];
                            M[it][rl] = xg[rl];
                        }
                        it++;
                    }
                }
                break;
            case 3:
                Gershgorin(xg, n);
                local_min(s3, 3);
                if (LB_piecewise_q(sbar, xg, Ni, n) < UB) {
                    LBk[it][1] = LB_piecewise_q(sbar, xg, Ni, n);
                    LBk[it][2] = 0;
                    for (int rl = 1; rl <= n; rl++) {
                        xl[it][rl] = sbar[rl];
                        M[it][rl] = xg[rl];
                    }
                    it++;
                }
                break;
            case 4:
                Kqmax(xg);
                local_min(s3, 5);
                if (LB_q(sbar, xg, K1) < UB) {
                    LBk[it][1] = LB_q(sbar, xg, K1);
                    LBk[it][2] = 0;
                    for (int rl = 1; rl <= n; rl++) {
                        xl[it][rl] = sbar[rl];
                        M[it][rl] = xg[rl];
                    }
                    it++;
                }
                break;
            case 5:
                Gershgorin(xg, n);
                convexification_gamma(xg, n);
                if (convex == 1) {
                    local_min(s3, 6);
                    if (LB_gamma(sbar, xg, gamma) < UB) {
                        LBk[it][1] = LB_p(sbar, xg, gamma);
                        LBk[it][2] = 0;
                        for (int rl = 1; rl <= n; rl++) {
                            xl[it][rl] = sbar[rl];
                            M[it][rl] = xg[rl];
                        }
                        it++;
                    }
                }
                else {
                    local_min(s3, 1);
                    if (LBalpha(sbar, xg, k) < UB) {
                        LBk[it][1] = LBalpha(sbar, xg, k);
                        LBk[it][2] = 0;
                        for (int rl = 1; rl <= n; rl++) {
                            xl[it][rl] = sbar[rl];
                            M[it][rl] = xg[rl];
                        }
                        it++;
                    }
                }
                break;
            }

        }
        LB = 100000000;
        int done = 0;
        for (l = 2; l < it; l++) {
            if (LBk[l][1] < LB && LBk[l][2] == 0) {
                done = l;
                LB = LBk[l][1];
                for (int rl = 1; rl <= n; rl++) {
                    s3[rl] = xl[l][rl];
                    T[rl] = M[l][rl];
                }
            }
            //std::cout << "LB in" << LB << '\n';
        }
        if (done != 0) { LBk[done][2] = 1; }

        if (UB - LB < eps || UB - LB == eps) {
            jend = it + 2;
        }

        iteration++;
    }

    std::cout << "\n La sollution optimale est \n\n";
    std::cout << "s1 opt=" << sopt[1] << '\n';
    std::cout << "s2 opt=" << sopt[2] << '\n';
    std::cout << "fmin" << UB << '\n';
    std::cout << "fopt" << fx(sopt) << '\n';
    std::cout << "iteration = " << iteration << "\n\n";
    
    /*
    for (int j = 1; j <= n; j++) {
        xg[j] = x[j];
        s3[j] = (xg[j].upper() + xg[j].lower()) / 2;
        sbar[j] = 2;
    }
    Kqmax(xg);
    //local_min(s3, 5);
    std::cout << "LBq= " << LB_q(sbar, xg, K1) << '\n';
    Gershgorin(x, n);
    //local_min(s3, 1);
    std::cout << "LBa= " << LBalpha(sbar, x, k) << '\n';
    convexification(x, n);
    //local_min(s3, 1);
    std::cout << "LBp= " << LBalpha(sbar, x, k) << '\n';
    //local_min(s3, 3);
    std::cout << "LB_piecewise_q= " << LB_piecewise_q(sbar, x, Ni, n) << '\n';
    //local_min(s3, 2);
    std::cout << "fx= " << fx(sbar) << '\n';
    */
}

