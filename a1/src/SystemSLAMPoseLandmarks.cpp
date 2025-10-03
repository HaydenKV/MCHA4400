#include <cmath>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"
#include "SystemSLAMPoseLandmarks.h"
#include <Eigen/Cholesky>

SystemSLAMPoseLandmarks::SystemSLAMPoseLandmarks(const GaussianInfo<double> & density)
    : SystemSLAM(density)
{

}

SystemSLAM * SystemSLAMPoseLandmarks::clone() const
{
    return new SystemSLAMPoseLandmarks(*this);
}

std::size_t SystemSLAMPoseLandmarks::numberLandmarks() const
{
    return (density.dim() - 12)/6;
}

std::size_t SystemSLAMPoseLandmarks::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 12 + 6*idxLandmark;    
}

std::size_t SystemSLAMPoseLandmarks::appendLandmark(
    const Eigen::Vector3d& rnj,
    const Eigen::Vector3d& Thetanj,
    const Eigen::Matrix<double,6,6>& Sj
)
{
    // Current state size
    const Eigen::VectorXd mu_old = density.mean();
    const Eigen::MatrixXd S_old  = density.sqrtCov();
    const int n_old = static_cast<int>(mu_old.size());

    // New sizes
    const int n_new = n_old + 6;

    // Build new mean
    Eigen::VectorXd mu_new(n_new);
    mu_new.head(n_old) = mu_old;
    mu_new.segment<3>(n_old + 0) = rnj;      // position
    mu_new.segment<3>(n_old + 3) = Thetanj;  // Euler rpy

    // Build new sqrt-cov (upper-triangular by construction)
    Eigen::MatrixXd S_new = Eigen::MatrixXd::Zero(n_new, n_new);
    // existing block
    S_new.topLeftCorner(n_old, n_old) = S_old;
    // new landmark block (assumed upper-triangular)
    S_new.block(n_old, n_old, 6, 6) = Sj;

    // IMPORTANT: We keep cross terms zero initially. If you want to inject camera
    // pose uncertainty and PnP uncertainty as cross-terms, this is where you would
    // compute them and fill the off-diagonal blocks.

    // Recreate the Gaussian from sqrt-moments
    density = GaussianInfo<double>::fromSqrtMoment(mu_new, S_new);

    // Return index of this landmark (0-based among landmarks, not absolute in the vector)
    // If landmarks are contiguous after 12 body states, landmark index is:
    const std::size_t lm_index = static_cast<std::size_t>((n_new - 12) / 6 - 1);
    return lm_index;
}