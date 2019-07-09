#include <iostream>
#include <vector>
#include <stdexcept>

#include "../Eigen/Dense"

#include "../tools.hpp"


using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;


namespace {

constexpr double kErrThreshold = 1e-6;

double CalculateError(const MatrixXd& a, const MatrixXd& b) {
    const auto residual = a - b;
    const auto dev = residual.array().pow(2);
    const double err = sqrt(dev.mean());
    return err;
}

void JacobiansTest1() {
    VectorXd x(4);
    x << 1, 2, 0.2, 0.4;

    MatrixXd Hj_sample(3, 4);
    Hj_sample << 0.447214, 0.894427, 0, 0,
                 -0.4, 0.2, 0, 0,
                 0, 0, 0.447214, 0.894427;

    const MatrixXd Hj = Tools::CalculateJacobian(x);

    const double err = CalculateError(Hj, Hj_sample);
    if (err > kErrThreshold) {
        std::cerr << "Jacobians have large error!" << std::endl;
        throw std::runtime_error(std::string(__FUNCTION__) + " Failed");
    }
}

void RMSETest1() {
    vector<VectorXd> xs;
    VectorXd x(4);
    x << 1, 1, 0.2, 0.1;
    xs.push_back(x);
    x << 2, 2, 0.3, 0.2;
    xs.push_back(x);
    x << 3, 3, 0.4, 0.3;
    xs.push_back(x);

    vector<VectorXd> ys;
    VectorXd y(4);
    y << 1.1, 1.1, 0.3, 0.2;
    ys.push_back(y);
    y << 2.1, 2.1, 0.4, 0.3;
    ys.push_back(y);
    y << 3.1, 3.1, 0.5, 0.4;
    ys.push_back(y);

    VectorXd rmse_sample(4);
    rmse_sample << 0.1, 0.1, 0.1, 0.1;

    const VectorXd rmse = Tools::CalculateRMSE(xs, ys);

    std::cout << rmse << std::endl;

    const double err = CalculateError(rmse, rmse_sample);
    if (err > kErrThreshold) {
        std::cerr << "RMSE have large error!" << std::endl;
        throw std::runtime_error(std::string(__FUNCTION__) + " Failed");
    }
}

}

int main() {
    JacobiansTest1();
    RMSETest1();

    return 0;
}

