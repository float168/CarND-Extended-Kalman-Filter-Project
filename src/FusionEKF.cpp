#include "FusionEKF.hpp"
#include <iostream>
#include "Eigen/Dense"
#include "tools.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

#define DEBUG_OUTPUT

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  MatrixXd P = MatrixXd(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1000, 0,
       0, 0, 0, 1000;

  MatrixXd F = MatrixXd(4, 4);
  F << 1, 0, 1, 0,
       0, 1, 0, 1,
       0, 0, 1, 0,
       0, 0, 0, 1;

  const VectorXd x = VectorXd::Zero(4);
  const MatrixXd Q = MatrixXd::Zero(4, 4);
  ekf_.Init(x, P, F, H_laser_, R_laser_, Q);

  noise_ax_ = 9;
  noise_ay_ = 9;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  const auto& measure = measurement_pack.raw_measurements_;
  const auto type = measurement_pack.sensor_type_;

  /**
   * Initialization
   */

  if (!is_initialized_) {
    // first measurement
    ekf_.x_ = VectorXd(4);

    switch (type) {
      case SensorType::RADAR:
      {
        ekf_.x_ = Tools::PolarToCartesian(measure);
        break;
      }
      case SensorType::LASER:
      {
        ekf_.x_ << measure[0], measure[1], 0, 0;
        break;
      }
      default:
        break;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  const double dt = 1e-6 * (measurement_pack.timestamp_ - previous_timestamp_);
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  const double dt2 = dt * dt;
  const double dt3_2 = dt2 * dt / 2.0;
  const double dt4_4 = dt3_2 * dt / 2.0;

  ekf_.Q_ << dt4_4 * noise_ax_, 0, dt3_2 * noise_ax_, 0,
             0, dt4_4 * noise_ay_, 0, dt3_2 * noise_ay_,
             dt3_2 * noise_ax_, 0, dt2 * noise_ax_, 0,
             0, dt3_2 * noise_ay_, 0, dt2 * noise_ay_;

  ekf_.Predict();

# ifdef DEBUG_OUTPUT
  cout << "=== measure ===" << endl;
  cout << "type: " << (type == SensorType::RADAR ? "Radar" : "Laser") << endl;
  cout << "z:" << endl << measure << endl;

  cout << "=== predict ===" << endl;
  cout << "x:" << endl << ekf_.x_ << endl;
  cout << "P:" << endl << ekf_.P_ << endl;
  cout << "F:" << endl << ekf_.F_ << endl;
  cout << "Q:" << endl << ekf_.Q_ << endl;
# endif

  /**
   * Update
   */

  if (type == SensorType::RADAR) {
    Hj_ = Tools::CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measure);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measure);
  }

# ifdef DEBUG_OUTPUT
  cout << "=== update ===" << endl;
  cout << "x:" << endl << ekf_.x_ << endl;
  cout << "P:" << endl << ekf_.P_ << endl;
  cout << "H:" << endl << ekf_.H_ << endl;
  cout << "R:" << endl << ekf_.R_ << endl;
# endif

  previous_timestamp_ = measurement_pack.timestamp_;
}
