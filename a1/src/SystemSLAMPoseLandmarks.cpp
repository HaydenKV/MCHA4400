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

std::size_t SystemSLAMPoseLandmarks::appendLandmark(const Eigen::Vector3d& r_nL,
                                                    const Eigen::Vector3d& Theta_nL,
                                                    const Eigen::Matrix<double,6,6>& Sj)
{
    // Current mean and sqrt-info
    const Eigen::VectorXd mu  = density.mean();
    const Eigen::MatrixXd Xi  = density.sqrtInfoMat();   // upper-triangular (Xi^T Xi = P^{-1})
    const int nx = static_cast<int>(mu.size());

    // Recover covariance P from Xi (solve twice for stability)
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nx, nx);
    Eigen::MatrixXd Y = Xi.transpose().triangularView<Eigen::Lower>().solve(I);
    Eigen::MatrixXd P = Xi.triangularView<Eigen::Upper>().solve(Y);

    // Get S (upper-triangular) with S^T S = P
    Eigen::LLT<Eigen::MatrixXd> llt(P);
    Eigen::MatrixXd Sold = llt.matrixL().transpose();

    // Augment mean: [mu; r; Theta]
    Eigen::VectorXd mu_aug(nx + 6);
    mu_aug << mu, r_nL, Theta_nL;

    // Augment sqrt-moment S_aug = blkdiag(Sold, Sj)
    Eigen::MatrixXd Saug = Eigen::MatrixXd::Zero(nx + 6, nx + 6);
    Saug.topLeftCorner(nx, nx) = Sold;
    Saug.bottomRightCorner(6, 6) = Sj;

    density = GaussianInfo<double>::fromSqrtMoment(mu_aug, Saug);

    // Return last landmark index
    return numberLandmarks() - 1;
}