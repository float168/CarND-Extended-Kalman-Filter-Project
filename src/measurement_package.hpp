#ifndef MEASUREMENT_PACKAGE_HPP_
#define MEASUREMENT_PACKAGE_HPP_

#include "Eigen/Dense"

enum class SensorType {
  LASER,
  RADAR,
};

class MeasurementPackage {
 public:
  SensorType sensor_type_;

  long long timestamp_;

  Eigen::VectorXd raw_measurements_;
};

#endif // MEASUREMENT_PACKAGE_HPP_
