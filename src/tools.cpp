#include "tools.hpp"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    const size_t size = estimations.size();

    if (size != ground_truth.size() || size == 0) {
        std::cerr << __FUNCTION__ << ": Invalid data" << std::endl;
        return rmse;
    }

    for (int i = 0; i < size; ++i) {
        const VectorXd residual = estimations[i] - ground_truth[i];
        const VectorXd deviation = residual.array().pow(2);
        rmse += deviation;
    }

    rmse = rmse / size;

    rmse = rmse.array().sqrt();

    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3, 4);
    Hj << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;

    const double px = x_state(0);
    const double py = x_state(1);
    const double vx = x_state(2);
    const double vy = x_state(3);

    const double d2 = px * px + py * py;

    if (fabs(d2) < 1e-6) {
        std::cerr << __FUNCTION__ << ": Devision by zero" << std::endl;
        return Hj;
    }

    const double d1 = sqrt(d2);
    const double px_d1 = px / d1;
    const double py_d1 = py / d1;
    const double det_d2 = (vx * py - vy * px) / d2;

    Hj << px_d1, py_d1, 0, 0,
          -py/d2, px/d2, 0, 0,
          py_d1 * det_d2, -px_d1 * det_d2, px_d1, py_d1;

    return Hj;
}

VectorXd Tools::CartesianToPolar(const VectorXd &x) {
    const double px = x[0];
    const double py = x[1];
    const double vx = x[2];
    const double vy = x[3];

    const double rho = sqrt(px * px + py * py);
    const double phi = atan2(py, px);
    const double rho_dot = fabs(rho) < 1e-6 ? 0.0 : (px * vx + py * vy) / rho;

    VectorXd polar(3);
    polar << rho, phi, rho_dot;
    return polar;
}

VectorXd Tools::PolarToCartesian(const VectorXd &x) {
    const double rho = x[0];
    const double phi = x[1];
    const double rhd = x[2];

    const double px = rho * cos(phi);
    const double py = rho * sin(phi);
    const double vx = rhd * sin(phi);
    const double vy = rhd * sin(phi);

    VectorXd cartesian(3);
    cartesian << px, py, vx, vy;
    return cartesian;
}

VectorXd Tools::NormalizePolar(const VectorXd &x) {
    VectorXd normed = x;

    double phi = normed[1];
    while (phi > M_PI) {
        phi -= 2 * M_PI;
    }
    while (phi <= -M_PI) {
        phi += 2 * M_PI;
    }
    normed(1) = phi;

    return normed;
}
