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

std::size_t SystemSLAMPointLandmarks::appendFromDuckDetections(const Camera& cam,
                                                               const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                                                               const Eigen::VectorXd& A,
                                                               double fx, double fy,
                                                               double duck_r_m,
                                                               double pos_sigma_m)
{
    // Prior-mean camera pose (rCNn, Rnc) from the filter state
    const Eigen::VectorXd xbar = density.mean();
    Eigen::MatrixXd Jdummy;
    const Eigen::Vector3d rCNn = SystemSLAM::cameraPosition(cam, xbar, Jdummy);
    const Eigen::Matrix3d Rnc  = SystemSLAM::cameraOrientation(cam, xbar);

    std::size_t nAdded = 0;

    for (int i = 0; i < Yuv.cols(); ++i)
    {
        // 1) Read centroid + area, guard tiny/NaN areas
        const double u = Yuv(0,i);
        const double v = Yuv(1,i);
        const double Ai = std::max(A(i), 1e-6);  // px^2 guard

        // 2) Depth from mask area model:
        //    A ≈ (fx*fy*π r^2) / Z^2  →  Z = sqrt((fx*fy*π r^2)/A)
        const double numer = fx * fy * std::numbers::pi * duck_r_m * duck_r_m;
        const double depth = std::sqrt(numer / Ai);
        if (!std::isfinite(depth) || depth <= 0.05) continue;

        // 3) Unit camera ray u_PC^c from pixel (u,v) via Camera (OpenCV types)
        //    pixelToVector returns a unit vector (z>0 if in front).
        const cv::Vec3d uPCc_cv = cam.pixelToVector(cv::Vec2d(u, v));
        if (!cam.isVectorWithinFOV(uPCc_cv)) continue;

        // 4) Scale ray to the inferred depth and convert to Eigen
        Eigen::Vector3d rPCc;
        cv::cv2eigen(uPCc_cv, rPCc);   // unit -> Eigen
        rPCc *= depth;                  // place point in camera coordinates

        // 5) Camera -> world: r^n_{L/N} = R_nc * r^c_{P/C} + r^n_{C/N}
        const Eigen::Vector3d rLNn = Rnc * rPCc + rCNn;

        // 6) Seed sqrt-covariance (std devs on the diagonal; upper-triangular)
        Eigen::Matrix3d Spos = Eigen::Matrix3d::Zero();
        Spos(0,0) = pos_sigma_m;
        Spos(1,1) = pos_sigma_m;
        Spos(2,2) = pos_sigma_m;

        appendLandmark(rLNn, Spos);
        ++nAdded;
    }
    return nAdded;
}
