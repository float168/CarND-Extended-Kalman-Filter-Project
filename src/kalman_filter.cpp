#include "kalman_filter.hpp"
#include "tools.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(const VectorXd &x_in, const MatrixXd &P_in, const MatrixXd &F_in,
                        const MatrixXd &H_in, const MatrixXd &R_in, const MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;

  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd Si = S.inverse();
  const MatrixXd K = P_ * Ht * Si;

  const VectorXd y = z - H_ * x_;
  x_ = x_ + K * y;

  const int size = P_.rows();
  P_ = (MatrixXd::Identity(size, size) - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd Si = S.inverse();
  const MatrixXd K = P_ * Ht * Si;

  const VectorXd hx = Tools::CartesianToPolar(x_);
  const VectorXd y = z - hx;
  x_ += K * y;

  const int size = P_.rows();
  P_ = (MatrixXd::Identity(size, size) - K * H_) * P_;
}
