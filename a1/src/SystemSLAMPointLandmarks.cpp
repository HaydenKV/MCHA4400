#include "SystemSLAMPointLandmarks.h"
#include <cassert>
#include <cmath>
#include <numbers>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

SystemSLAMPointLandmarks::SystemSLAMPointLandmarks(const GaussianInfo<double> & density)
    : SystemSLAM(density)
{}

SystemSLAM * SystemSLAMPointLandmarks::clone() const
{
    return new SystemSLAMPointLandmarks(*this);
}

std::size_t SystemSLAMPointLandmarks::numberLandmarks() const
{
    return (density.dim() - 12) / 3;
}

std::size_t SystemSLAMPointLandmarks::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 12 + 3*idxLandmark;
}

std::size_t SystemSLAMPointLandmarks::appendLandmark(const Eigen::Vector3d& rLNn,
                                                     const Eigen::Matrix3d& Spos)
{
    auto p_new = GaussianInfo<double>::fromSqrtMoment(rLNn, Spos);
    density = density * p_new;
    failCount_.push_back(0);
    return numberLandmarks() - 1;
}

std::size_t SystemSLAMPointLandmarks::appendFromDuckDetections(const Camera& cam,
                                                               const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                                                               const Eigen::VectorXd& A,
                                                               double fx, double fy,
                                                               double duck_r_m,
                                                               double pos_sigma_m)
{
    const Eigen::VectorXd xbar = density.mean();
    const Pose<double> Tnc = cam.bodyToCamera(Pose<double>(SystemSLAM::cameraOrientation(cam, xbar), SystemSLAM::cameraPosition(cam, xbar)));
    const Eigen::Matrix3d Rnc = Tnc.rotationMatrix;
    const Eigen::Vector3d rCNn = Tnc.translationVector;

    std::size_t nAdded = 0;
    for (int i = 0; i < Yuv.cols(); ++i)
    {
        const double u = Yuv(0,i);
        const double v = Yuv(1,i);
        const double Ai = std::max(A(i), 1e-6);

        const double depth_squared = (fx * fy * std::numbers::pi * duck_r_m * duck_r_m) / Ai;
        if (depth_squared <= 0.0) continue;
        const double depth = std::sqrt(depth_squared);

        const cv::Vec3d uPCc_cv = cam.pixelToVector(cv::Vec2d(u, v));
        if (!cam.isVectorWithinFOVConservative(uPCc_cv)) continue;

        Eigen::Vector3d uPCc_eigen;
        cv::cv2eigen(uPCc_cv, uPCc_eigen);

        const Eigen::Vector3d rPCc = depth * uPCc_eigen;
        const Eigen::Vector3d rLNn = Rnc * rPCc + rCNn;

        Eigen::Matrix3d Spos = Eigen::Matrix3d::Identity() * pos_sigma_m;

        appendLandmark(rLNn, Spos);
        ++nAdded;
    }
    return nAdded;
}

void SystemSLAMPointLandmarks::updateFailureCounter(std::size_t j, bool incrementIfFailed)
{
    if (j >= failCount_.size()) failCount_.resize(numberLandmarks(), 0);
    if (incrementIfFailed) ++failCount_[j];
    else                   failCount_[j] = 0;
}

void SystemSLAMPointLandmarks::cullFailed(int threshold)
{
    if (failCount_.empty()) return;
    if ((int)failCount_.size() < (int)numberLandmarks())
        failCount_.resize(numberLandmarks(), 0);

    // build “keep” mask per landmark
    std::vector<char> keepLM(numberLandmarks(), 1);
    bool anyDelete = false;
    for (std::size_t j = 0; j < numberLandmarks(); ++j) {
        if (failCount_[j] >= threshold) {
            keepLM[j] = 0;
            anyDelete = true;
        }
    }
    if (!anyDelete) return;

    // build list of state indices to KEEP: body (0..11) + 3 per kept landmark
    std::vector<std::size_t> keepIdx;
    keepIdx.reserve(12 + 3*numberLandmarks());
    for (std::size_t k = 0; k < 12; ++k) keepIdx.push_back(k);
    for (std::size_t j = 0; j < numberLandmarks(); ++j) {
        if (!keepLM[j]) continue;
        const std::size_t base = 12 + 3*j;
        keepIdx.push_back(base + 0);
        keepIdx.push_back(base + 1);
        keepIdx.push_back(base + 2);
    }

    // remove (marginalise out) the failed landmarks
    density = density.marginal(keepIdx);

    // compact failure counters
    std::vector<int> newFail;
    newFail.reserve(numberLandmarks()); // numberLandmarks() now reflects new size
    for (std::size_t j = 0; j < keepLM.size(); ++j)
        if (keepLM[j]) newFail.push_back(failCount_[j]);
    failCount_.swap(newFail);
}
