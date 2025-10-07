#include <cmath>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "Pose.hpp"
#include <cassert>
#include "SystemSLAM.h"
#include "SystemSLAMPointLandmarks.h"

SystemSLAMPointLandmarks::SystemSLAMPointLandmarks(const GaussianInfo<double> & density)
    : SystemSLAM(density)
{

}

SystemSLAM * SystemSLAMPointLandmarks::clone() const
{
    return new SystemSLAMPointLandmarks(*this);
}

std::size_t SystemSLAMPointLandmarks::numberLandmarks() const
{
    return (density.dim() - 12)/3;
}

std::size_t SystemSLAMPointLandmarks::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 12 + 3*idxLandmark;    
}


// ********** NEW: append one 3-D point landmark **********
std::size_t SystemSLAMPointLandmarks::appendLandmark(const Eigen::Vector3d& rLNn,
                                                     const Eigen::Matrix3d& Spos)
{
    // Build a 3D Gaussian for the new landmark and "independently" augment the joint:
    // new_joint(x, m_new) = old_joint(x) * N(rLNn, Spos)
    auto p_new = GaussianInfo<double>::fromSqrtMoment(rLNn, Spos);
    density = density * p_new;

    // Return the new landmark index (last one now)
    return numberLandmarks() - 1;
}

// (your existing function, unchanged except it now calls appendLandmark)
void SystemSLAMPointLandmarks::appendFromDuckDetections(
    const Camera& camera,
    const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
    const Eigen::VectorXd& A,
    double fx, double fy,
    double duck_r_m,
    double sigma_pos_m)
{
    assert(Yuv.cols() == A.size());

    // Current mean pose
    const Eigen::VectorXd xmean = density.mean();
    const Eigen::Vector3d rCNn  = cameraPosition(camera, xmean);
    const Eigen::Matrix3d Rnc   = cameraOrientation(camera, xmean);

    const double cx = camera.cameraMatrix.at<double>(0,2);
    const double cy = camera.cameraMatrix.at<double>(1,2);

    for (int i = 0; i < Yuv.cols(); ++i)
    {
        const double Af = A(i);
        if (Af <= 0.0) continue;

        // 1) Range from apparent area model (A ∝ 1/d^2)
        const double d = std::sqrt((fx*fy) * M_PI * duck_r_m * duck_r_m / Af);

        // 2) Ray in camera frame
        const double u = Yuv(0,i), v = Yuv(1,i);
        Eigen::Vector3d dir_c((u - cx)/fx, (v - cy)/fy, 1.0);
        dir_c.normalize();

        // 3) Landmark in nav frame
        const Eigen::Vector3d rLNn = rCNn + Rnc * (d * dir_c);

        // 4) Conservative 3×3 sqrt-covariance
        Eigen::Matrix3d Spos = sigma_pos_m * Eigen::Matrix3d::Identity();

        appendLandmark(rLNn, Spos);
    }
}
