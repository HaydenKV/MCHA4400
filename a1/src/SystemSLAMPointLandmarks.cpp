#include <cmath>
#include <numbers>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

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

// ********** Append one 3-D point landmark **********
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

// CORRECTED AND IMPROVED FUNCTION
std::size_t SystemSLAMPointLandmarks::appendFromDuckDetections(const Camera& cam,
                                                               const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                                                               const Eigen::VectorXd& A,
                                                               double fx, double fy,
                                                               double duck_r_m,
                                                               double pos_sigma_m)
{
    // Get the prior-mean camera pose (Tnc), which defines the camera's position (rCNn) and orientation (Rnc) in the world frame.
    // This is the reference from which we will triangulate the new landmark positions.
    const Eigen::VectorXd xbar = density.mean();
    const Pose<double> Tnc = cam.bodyToCamera(Pose<double>(SystemSLAM::cameraOrientation(cam, xbar), SystemSLAM::cameraPosition(cam, xbar)));
    const Eigen::Matrix3d Rnc = Tnc.rotationMatrix;
    const Eigen::Vector3d rCNn = Tnc.translationVector;

    std::size_t nAdded = 0;

    for (int i = 0; i < Yuv.cols(); ++i)
    {
        const double u = Yuv(0,i);
        const double v = Yuv(1,i);
        const double Ai = std::max(A(i), 1e-6);  // Guard against division by zero or negative area.

        // 1. Estimate depth from the mask area using the inverse-square law, as per the assignment.
        // A ≈ (fx*fy*π*r²) / Z²  =>  Z = sqrt((fx*fy*π*r²)/A)
        const double depth_squared = (fx * fy * std::numbers::pi * duck_r_m * duck_r_m) / Ai;
        if (depth_squared <= 0.0) continue; // Skip if area is non-physical.
        const double depth = std::sqrt(depth_squared);

        // 2. Back-project the pixel centroid (u,v) to a 3D unit vector in the camera's coordinate frame.
        // The pixelToVector function correctly handles the camera's intrinsics and distortion.
        const cv::Vec3d uPCc_cv = cam.pixelToVector(cv::Vec2d(u, v));
        Eigen::Vector3d uPCc_eigen;
        cv::cv2eigen(uPCc_cv, uPCc_eigen);

        // 3. (IMPROVEMENT) Use the more robust 'isVectorWithinFOVConservative' check.
        // This ensures the landmark is not initialized right at the image edge, which can be unstable.
        if (!cam.isVectorWithinFOVConservative(uPCc_cv)) continue;

        // 4. Scale the unit vector by the estimated depth to get the landmark's position relative to the camera.
        const Eigen::Vector3d rPCc = depth * uPCc_eigen;

        // 5. Transform the 3D point from the camera frame to the world frame to get the landmark's world position.
        // r_LN_n = R_nc * r_PC_c + r_CN_n
        const Eigen::Vector3d rLNn = Rnc * rPCc + rCNn;

        // 6. Create an initial diagonal covariance for the new landmark.
        Eigen::Matrix3d Spos = Eigen::Matrix3d::Identity() * pos_sigma_m;

        // 7. Append the new landmark to the state vector.
        appendLandmark(rLNn, Spos);
        ++nAdded;
    }
    return nAdded;
}