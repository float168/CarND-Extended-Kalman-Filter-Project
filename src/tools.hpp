#ifndef TOOLS_HPP_
#define TOOLS_HPP_

#include <vector>
#include "Eigen/Dense"

class Tools {
 public:
  /**
   * Constructor.
   */
  Tools();

  /**
   * Destructor.
   */
  virtual ~Tools();

  /**
   * A helper method to calculate RMSE.
   */
  static Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, 
                                const std::vector<Eigen::VectorXd> &ground_truth);

  /**
   * A helper method to calculate Jacobians.
   */
  static Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);

  /**
   * A helper method to convert cartesian coord to polar coord.
   */
  static Eigen::VectorXd CartesianToPolar(const Eigen::VectorXd &x);

  /**
   * A helper method to convert polar coord to cartesian coord.
   */
  static Eigen::VectorXd PolarToCartesian(const Eigen::VectorXd &x);

  /**
   * A helper method to normalize radian expression of polar coord vector
   */
  static Eigen::VectorXd NormalizePolar(const Eigen::VectorXd &x);
};

#endif  // TOOLS_HPP_
