#include <cassert>
#include <numeric>
#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "GaussianInfo.hpp"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "SystemSLAM.h"
#include "SystemSLAMPoseLandmarks.h"
#include "Camera.h"
#include "Pose.hpp"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using autodiff::dual;
using autodiff::jacobian;
using autodiff::wrt;
using autodiff::at;
using autodiff::val;

MeasurementSLAMUniqueTagBundle::MeasurementSLAMUniqueTagBundle(
    double time,
    const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
    const Camera& camera,
    const std::vector<TagDetection>& detections,
    double tagSizeMeters)
: MeasurementSLAM(time, camera)
, Y_(Y)
, detections_(detections)
, tagSize_(tagSizeMeters)
{
    // Visualisation-friendly default
    // updateMethod_ etc. can be set when you enable fusion
    idxFeatures_.assign(0, 0);
}

template <typename Scalar>
std::array<Eigen::Vector3<Scalar>, 4>
MeasurementSLAMUniqueTagBundle::tagCornersWorld(const Eigen::Vector3<Scalar>& rLNn,
                                                const Eigen::Matrix3<Scalar>& Rln) const
{
    const Scalar s = static_cast<Scalar>(0.5 * tagSize_);
    // Tag plane is the landmark's x–y plane; z-axis is tag normal
    std::array<Eigen::Vector3<Scalar>, 4> P;
    // TL, TR, BR, BL in the tag frame
    Eigen::Matrix<Scalar,3,4> rQLl;
    rQLl.col(0) << -s,  s, 0;  // TL
    rQLl.col(1) <<  s,  s, 0;  // TR
    rQLl.col(2) <<  s, -s, 0;  // BR
    rQLl.col(3) << -s, -s, 0;  // BL
    for (int i = 0; i < 4; ++i)
        P[i] = rLNn + Rln * rQLl.col(i);
    return P;
}

template <typename Scalar>
Eigen::Vector2<Scalar>
MeasurementSLAMUniqueTagBundle::predictTagCenter(const Eigen::VectorX<Scalar>& x,
                                                 const SystemSLAM& system,
                                                 std::size_t idxPoseLandmark) const
{
    // Body pose
    Pose<Scalar> Tnb;
    Tnb.rotationMatrix   = rpy2rot(x.template segment<3>(9)); // Rnb
    Tnb.translationVector= x.template segment<3>(6);          // rBNn
    Pose<Scalar> Tnc = camera_.bodyToCamera(Tnb);             // camera pose

    // Landmark pose block: [rLNn(3); Thetaln(3)]
    std::size_t idx = system.landmarkPositionIndex(idxPoseLandmark);
    Eigen::Vector3<Scalar> rLNn = x.template segment<3>(idx);
    Eigen::Vector3<Scalar> Thetaln = x.template segment<3>(idx + 3);
    Eigen::Matrix3<Scalar> Rln = rpy2rot(Thetaln);

    // Center of tag in world = rLNn (we store the center)
    const Eigen::Matrix3<Scalar> Rcn = Tnc.rotationMatrix.transpose();
    Eigen::Vector3<Scalar> rPCc = Rcn * (rLNn - Tnc.translationVector);
    return camera_.vectorToPixel(rPCc);
}

// --- required virtuals for Plot / framework ---

GaussianInfo<double>
MeasurementSLAMUniqueTagBundle::predictFeatureDensity(const SystemSLAM& system, std::size_t idxLandmark) const
{
    const std::size_t nx = system.density.dim();
    const std::size_t ny = 2;

    auto func = [&](const Eigen::VectorXd& xv, Eigen::MatrixXd& Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::Vector2d v = xv.tail<2>();

        // autodiff center projection for landmark idxLandmark
        Eigen::VectorX<dual> xad = x.cast<dual>();
        Eigen::Vector2<dual> ydual;
        Eigen::MatrixXd Jx;
        // wrap as function of x for jacobian
        auto h = [&](const Eigen::VectorX<dual>& xin)->Eigen::Vector2<dual>{
            return predictTagCenter<dual>(xin, system, idxLandmark);
        };
        Jx = jacobian(h, wrt(xad), at(xad), ydual);
        Eigen::Vector2d y; y << val(ydual(0)), val(ydual(1));
        Ja.resize(ny, nx + ny);
        Ja << Jx, Eigen::Matrix2d::Identity();
        return y + v;
    };

    auto pv  = GaussianInfo<double>::fromSqrtMoment(sigma_ * Eigen::Matrix2d::Identity());
    auto pxv = system.density * pv;
    return pxv.affineTransform(func);
}

GaussianInfo<double>
MeasurementSLAMUniqueTagBundle::predictFeatureBundleDensity(const SystemSLAM& system,
                                                            const std::vector<std::size_t>& idxLandmarks) const
{
    const std::size_t nx = system.density.dim();
    const std::size_t ny = 2 * idxLandmarks.size();

    auto func = [&](const Eigen::VectorXd& xv, Eigen::MatrixXd& Ja)
    {
        assert(xv.size() == nx + ny);
        const Eigen::VectorXd x = xv.head(nx);
        const Eigen::VectorXd v = xv.tail(ny);

        // stack centers
        Eigen::VectorXd h(ny);
        Eigen::MatrixXd J(ny, nx);
        J.setZero();

        for (std::size_t i = 0; i < idxLandmarks.size(); ++i)
        {
            // autodiff center for each
            Eigen::VectorX<dual> xad = x.cast<dual>();
            Eigen::Vector2<dual> ydual;
            Eigen::MatrixXd Jx;
            auto h1 = [&](const Eigen::VectorX<dual>& xin)->Eigen::Vector2<dual>{
                return predictTagCenter<dual>(xin, system, idxLandmarks[i]);
            };
            Jx = jacobian(h1, wrt(xad), at(xad), ydual);
            h.segment<2>(2*i) << val(ydual(0)), val(ydual(1));
            J.block(2*i, 0, 2, nx) = Jx;
        }

        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return h + v;
    };

    auto pv  = GaussianInfo<double>::fromSqrtMoment(sigma_ * Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density * pv;
    return pxv.affineTransform(func);
}

const std::vector<int>&
MeasurementSLAMUniqueTagBundle::associate(const SystemSLAM& /*system*/,
                                          const std::vector<std::size_t>& idxLandmarks)
{
    // Stage-1 stub: return "no association"
    idxFeatures_.assign(idxLandmarks.size(), -1);
    return idxFeatures_;
}

void MeasurementSLAMUniqueTagBundle::update(SystemBase& system)
{
    // Stage-1: skip map management & fusion — just call base to keep plumbing OK if needed later
    // (When enabling fusion, perform ID→landmark management here, then call Measurement::update(system))
    // Measurement::update(system);
    (void)system; // suppress unused warning for now
}

Eigen::VectorXd MeasurementSLAMUniqueTagBundle::simulate(const Eigen::VectorXd&,
                                                         const SystemEstimator&) const
{
    // Return a vector the same size as Y_ (column-stacked u,v)
    Eigen::VectorXd y(Y_.size());
    if (y.size() > 0) y.setZero();
    return y;
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(const Eigen::VectorXd&,
                                                     const SystemEstimator&) const
{
    // Stage-1: neutral score (no fusion yet)
    return 0.0;
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(const Eigen::VectorXd& x,
                                                     const SystemEstimator& sys,
                                                     Eigen::VectorXd& g) const
{
    g.resize(x.size());
    g.setZero();
    return logLikelihood(x, sys);
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(const Eigen::VectorXd& x,
                                                     const SystemEstimator& sys,
                                                     Eigen::VectorXd& g,
                                                     Eigen::MatrixXd& H) const
{
    g.resize(x.size()); g.setZero();
    H.resize(x.size(), x.size()); H.setZero();
    return logLikelihood(x, sys, g);
}
