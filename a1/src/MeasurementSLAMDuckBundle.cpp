#include "MeasurementSLAMDuckBundle.h"

#include <cassert>
#include <numeric>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include "SystemEstimator.h"
#include "SystemBase.h"
#include "Pose.hpp"
#include "association_util.h"

// -----------------------------------------------------------------------------
// Constructor
//  * Mirrors PointBundle’s behaviour: set method, stash measurements + params.
//  * Asserts ensure we always get consistent inputs (your request: no try/catch).
// -----------------------------------------------------------------------------
MeasurementSLAMDuckBundle::MeasurementSLAMDuckBundle(double time,
                                                     const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                                                     const Eigen::VectorXd& A,
                                                     const Camera& camera,
                                                     double sigma_c_px,
                                                     double sigma_a_px2,
                                                     double duck_radius_m)
: MeasurementSLAM(time, camera)
, Yuv_(Yuv)
, A_(A)
, sigma_c_(sigma_c_px)
, sigma_a_(sigma_a_px2)
, duck_r_m_(duck_radius_m)
, fx_(camera.cameraMatrix.at<double>(0,0))
, fy_(camera.cameraMatrix.at<double>(1,1))
{
    assert(Yuv_.cols() == A_.size() && "DuckBundle: Y(2xN) and A(N) must match");
    assert(sigma_c_ > 0.0 && sigma_a_ > 0.0 && duck_r_m_ > 0.0);
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM* MeasurementSLAMDuckBundle::clone() const
{
    // Deep-copy including association indices so Plot can mirror what happened
    auto* m = new MeasurementSLAMDuckBundle(time_, Yuv_, A_, camera_,
                                            sigma_c_, sigma_a_, duck_r_m_);
    m->idxFeatures_ = idxFeatures_;
    return m;
}

// -----------------------------------------------------------------------------
// predictDuckT<T> : core measurement model per landmark
//
// Matches the assignment model used in Lab 3/9 notes:
//   1) Compute camera pose from state, landmark position from state
//   2) Pixel centroid (u,v) via pinhole projection: camera_.vectorToPixel
//   3) Area A via inverse-square distance: A = (fx*fy*pi*r^2)/||rC - rL||^2
//
// We template on Scalar to get autodiff Jacobians with minimal code duplication.
// -----------------------------------------------------------------------------
template<typename Scalar>
Eigen::Matrix<Scalar,3,1>
MeasurementSLAMDuckBundle::predictDuckT(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
                                        const SystemSLAM& system,
                                        std::size_t idxLandmark) const
{
    // Camera pose from state (use the same helpers as PointBundle for parity)
    // r^n_{C/N}, R^n_c
    Pose<Scalar> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x);
    Tnc.rotationMatrix    = system.cameraOrientation(camera_, x);

    // Landmark position in nav frame
    const std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Matrix<Scalar,3,1> rLNn = x.template segment<3>(idx);

    // Project to pixels
    const Eigen::Matrix<Scalar,3,3> Rcn = Tnc.rotationMatrix.transpose();
    const Eigen::Matrix<Scalar,3,1> rLCc = Rcn * (rLNn - Tnc.translationVector);
    const Eigen::Matrix<Scalar,2,1> uv   = camera_.vectorToPixel(rLCc);

    // Inverse-square apparent area (distance in nav is fine since it's Euclidean)
    const Scalar dist2 = (Tnc.translationVector - rLNn).squaredNorm();
    const Scalar A     = Scalar(fx_*fy_) * Scalar(M_PI) * Scalar(duck_r_m_*duck_r_m_) / dist2;

    Eigen::Matrix<Scalar,3,1> h;
    h << uv(0), uv(1), A;
    return h;
}

// Explicit instantiations (keep linakge simple)
template Eigen::Matrix<double,3,1>
MeasurementSLAMDuckBundle::predictDuckT<double>(const Eigen::Matrix<double, Eigen::Dynamic, 1>&,
                                                const SystemSLAM&, std::size_t) const;
template Eigen::Matrix<autodiff::dual,3,1>
MeasurementSLAMDuckBundle::predictDuckT<autodiff::dual>(const Eigen::Matrix<autodiff::dual, Eigen::Dynamic, 1>&,
                                                        const SystemSLAM&, std::size_t) const;

// Non-templated front-end that also returns the Jacobian via autodiff
Eigen::Vector3d
MeasurementSLAMDuckBundle::predictDuck(const Eigen::VectorXd& x,
                                       Eigen::MatrixXd& J,
                                       const SystemSLAM& system,
                                       std::size_t idxLandmark) const
{
    using autodiff::dual;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;
    using autodiff::val;

    Eigen::Matrix<dual, Eigen::Dynamic, 1> xdual = x.cast<dual>();
    auto hfun = [&](const Eigen::Matrix<dual, Eigen::Dynamic, 1>& xd)->Eigen::Matrix<dual,3,1>
    {
        return predictDuckT<dual>(xd, system, idxLandmark);
    };

    Eigen::Matrix<dual,3,1> hdual;
    J = jacobian(hfun, wrt(xdual), at(xdual), hdual); // (3 x nx)

    Eigen::Vector3d h;
    h << val(hdual(0)), val(hdual(1)), val(hdual(2));
    return h;
}

// Bundle stacking helper
Eigen::VectorXd
MeasurementSLAMDuckBundle::predictDuckBundle(const Eigen::VectorXd& x,
                                             Eigen::MatrixXd& J,
                                             const SystemSLAM& system,
                                             const std::vector<std::size_t>& idxLandmarks) const
{
    const std::size_t nL = idxLandmarks.size();
    const std::size_t nx = system.density.dim();
    assert(x.size() == nx);

    Eigen::VectorXd h(3*nL);
    J.resize(3*nL, nx);

    for (std::size_t i = 0; i < nL; ++i) {
        Eigen::MatrixXd Ji;
        Eigen::Vector3d hi = predictDuck(x, Ji, system, idxLandmarks[i]);
        h.segment<3>(3*i) = hi;
        J.middleRows(3*i, 3) = Ji;
    }
    return h;
}

// -----------------------------------------------------------------------------
// Densities
//  * predictDuckBundleDensity: full [u,v,A] noise (block-diagonal per duck)
//  * predictFeatureDensity / predictFeatureBundleDensity:
//      These are required by Plot (left pane draws 2D ellipses). We return
//      densities **only for (u,v)** with sigma_c_, ignoring area, to match
//      PointBundle’s semantics that Plot expects.
// -----------------------------------------------------------------------------
GaussianInfo<double>
MeasurementSLAMDuckBundle::predictDuckBundleDensity(const SystemSLAM& system,
                                                    const std::vector<std::size_t>& idxLandmarks) const
{
    const std::size_t nx = system.density.dim();
    const std::size_t ny = 3 * idxLandmarks.size();

    // y_a = h(x) + v,  Ja = [ ∂h/∂x  I ]
    const auto func = [&](const Eigen::VectorXd& xv, Eigen::MatrixXd& Ja)
    {
        assert(xv.size() == nx + ny);
        const Eigen::VectorXd x = xv.head(nx);
        const Eigen::VectorXd v = xv.tail(ny);

        Eigen::MatrixXd Jx;
        Eigen::VectorXd hx = predictDuckBundle(x, Jx, system, idxLandmarks);

        Ja.resize(ny, nx + ny);
        Ja << Jx, Eigen::MatrixXd::Identity(ny, ny);
        return hx + v;
    };

    // Sqrt-cov S is block-diag per duck: diag([σc,σc,σa] ... )
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(ny, ny);
    for (std::size_t i = 0; i < idxLandmarks.size(); ++i) {
        const std::size_t r = 3*i;
        S(r+0, r+0) = sigma_c_;
        S(r+1, r+1) = sigma_c_;
        S(r+2, r+2) = sigma_a_;
    }

    auto pv  = GaussianInfo<double>::fromSqrtMoment(S);
    auto pxv = system.density * pv;  // independent noise
    return pxv.affineTransform(func);
}

// Centroid-only density (2D per feature) used for SNN association
GaussianInfo<double> MeasurementSLAMDuckBundle::predictCentroidBundleDensity(
    const SystemSLAM& system,
    const std::vector<std::size_t>& idxLandmarks) const
{
    const std::size_t nx = system.density.dim();
    const std::size_t L  = idxLandmarks.size();
    const std::size_t ny = 2 * L; // (u,v) per landmark

    const auto func = [&](const Eigen::VectorXd& xv, Eigen::MatrixXd& Ja)
    {
        assert(xv.size() == nx + ny);
        const Eigen::VectorXd x = xv.head(nx);
        const Eigen::VectorXd v = xv.tail(ny);

        // Build h2D and J2D by selecting u,v rows from the 3D (u,v,A) prediction.
        Eigen::VectorXd h2D(ny);
        Eigen::MatrixXd J2D(ny, nx);

        for (std::size_t i = 0; i < L; ++i)
        {
            Eigen::MatrixXd Ji;                   // (3×nx)
            Eigen::Vector3d hi = predictDuck(x, Ji, system, idxLandmarks[i]);
            // Copy (u,v)
            h2D.segment<2>(2*i) = hi.head<2>();
            J2D.block(2*i, 0, 2, nx) = Ji.topRows(2);
        }

        Ja.resize(ny, nx + ny);
        Ja << J2D, Eigen::MatrixXd::Identity(ny, ny);
        return h2D + v;
    };

    // Sqrt-noise: sigma_c_ on (u,v), per landmark (block diagonal)
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(ny, ny);
    for (std::size_t i = 0; i < L; ++i) {
        S(2*i + 0, 2*i + 0) = sigma_c_;
        S(2*i + 1, 2*i + 1) = sigma_c_;
    }

    auto pv  = GaussianInfo<double>::fromSqrtMoment(S);
    auto pxv = system.density * pv;   // p(x,v) = p(x) p(v), independent noise
    return pxv.affineTransform(func);
}

// Plot hooks: return 2D densities so left pane can draw ellipses at centers.
GaussianInfo<double>
MeasurementSLAMDuckBundle::predictFeatureDensity(const SystemSLAM& system,
                                                 std::size_t idxLandmark) const
{
    std::vector<std::size_t> one{idxLandmark};
    return predictCentroidBundleDensity(system, one);
}

GaussianInfo<double>
MeasurementSLAMDuckBundle::predictFeatureBundleDensity(const SystemSLAM& system,
                                                       const std::vector<std::size_t>& idxLandmarks) const
{
    return predictCentroidBundleDensity(system, idxLandmarks);
}

// -----------------------------------------------------------------------------
// Association
//  * Exactly like PointBundle, but pass CENTROID-ONLY density (2D) into SNN.
//  * Measurements Yuv_ are centroids only; area is excluded from SNN on purpose.
// -----------------------------------------------------------------------------
const std::vector<int>& MeasurementSLAMDuckBundle::associate(
    const SystemSLAM& system,
    const std::vector<std::size_t>& idxLandmarks)
{
    const std::size_t nL = system.numberLandmarks();

    // Reset per-frame flags and map size
    is_effectively_associated_.assign(nL, false);
    idxFeatures_.assign(nL, -1);

    if (idxLandmarks.empty() || Y().cols() == 0)
        return idxFeatures_;

    // Build centroid-only predicted density (2× per landmark)
    GaussianInfo<double> y2D = predictCentroidBundleDensity(system, idxLandmarks);

    // Standard SNN data association on (u,v)
    snn(system, y2D, idxLandmarks, Y(), camera_, idxFeatures_);

    // Mark which landmarks were matched this frame (for Plot colouring)
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j)
        if (idxFeatures_[j] >= 0) is_effectively_associated_[j] = true;

    return idxFeatures_;
}


// -----------------------------------------------------------------------------
// Update
//  * Keep identical flow to MeasurementPointBundle:
//      - build idxLandmarks in map order
//      - call associate(...) to fill idxFeatures_
//      - call base Measurement::update(system) to run the chosen optimiser
// -----------------------------------------------------------------------------
void MeasurementSLAMDuckBundle::update(SystemBase& systemBase)
{
    SystemSLAM& sys = dynamic_cast<SystemSLAM&>(systemBase);

    std::vector<std::size_t> idxLandmarks(sys.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);

    // Make sure vectors are sized even if nothing to match this frame
    is_effectively_associated_.assign(sys.numberLandmarks(), false);
    idxFeatures_.assign(sys.numberLandmarks(), -1);

    if (!idxLandmarks.empty() && Y().cols() > 0)
        associate(sys, idxLandmarks);

    // Perform the actual EKF/GN update via the Measurement base
    Measurement::update(systemBase);
}

// -----------------------------------------------------------------------------
// Likelihoods
//  * Weighted per-component with diag R^{-1} using σc for u,v and σa for A
//  * We follow the same 3 overload pattern as PointBundle to support
//    value-only, value+grad, value+grad+Hessian (GN) in the optimiser.
// -----------------------------------------------------------------------------
static inline void buildInvWeights(std::size_t k, double sc, double sa, Eigen::VectorXd& w)
{
    w.resize(3 * k);
    const double ic = 1.0 / (sc * sc);
    const double ia = 1.0 / (sa * sa);
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t r = 3 * i;
        w(r+0) = ic;
        w(r+1) = ic;
        w(r+2) = ia;
    }
}

Eigen::VectorXd
MeasurementSLAMDuckBundle::simulate(const Eigen::VectorXd& x,
                                    const SystemEstimator& system) const
{
    // Used by diagnostics / unit tests: simply return the stacked prediction
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    std::vector<std::size_t> idxLandmarks(sys.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);

    Eigen::MatrixXd J;
    return predictDuckBundle(x, J, sys, idxLandmarks);
}

double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    // Select paired (landmark j, feature f) from idxFeatures_
    std::vector<std::size_t> useL;
    std::vector<int> useF;
    useL.reserve(sys.numberLandmarks());
    useF.reserve(sys.numberLandmarks());

    const int N = static_cast<int>(A_.size());
    for (std::size_t j = 0; j < sys.numberLandmarks(); ++j) {
        if (j < idxFeatures_.size()) {
            const int f = idxFeatures_[j];
            if (f >= 0 && f < N) { useL.push_back(j); useF.push_back(f); }
        }
    }
    const std::size_t k = useL.size();
    if (k == 0) return 0.0;

    // Stack y = [u1 v1 A1 u2 v2 A2 ...]^T
    Eigen::VectorXd y(3 * k);
    for (std::size_t i = 0; i < k; ++i) {
        const int f = useF[i];
        y.segment<2>(3*i) = Yuv_.col(f);
        y(3*i + 2)        = A_(f);
    }

    Eigen::MatrixXd Jx;                          // not needed here
    Eigen::VectorXd h = predictDuckBundle(x, Jx, sys, useL);

    Eigen::VectorXd w;
    buildInvWeights(k, sigma_c_, sigma_a_, w);

    const Eigen::VectorXd r = y - h;

    double ll = 0.0;
    for (int i = 0; i < r.size(); ++i) ll += -0.5 * w(i) * r(i) * r(i); // drop const
    return ll;
}

double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system,
                                         Eigen::VectorXd& g) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    // Assemble matched subsets (same as above)
    std::vector<std::size_t> useL;
    std::vector<int> useF;
    const int N = static_cast<int>(A_.size());
    for (std::size_t j = 0; j < sys.numberLandmarks(); ++j) {
        if (j < idxFeatures_.size()) {
            const int f = idxFeatures_[j];
            if (f >= 0 && f < N) { useL.push_back(j); useF.push_back(f); }
        }
    }
    const std::size_t k = useL.size();

    g.setZero(x.size());
    if (k > 0)
    {
        Eigen::VectorXd y(3 * k);
        for (std::size_t i = 0; i < k; ++i) {
            const int f = useF[i];
            y.segment<2>(3*i) = Yuv_.col(f);
            y(3*i + 2)        = A_(f);
        }

        Eigen::MatrixXd J;
        Eigen::VectorXd h = predictDuckBundle(x, J, sys, useL);

        Eigen::VectorXd w;
        buildInvWeights(k, sigma_c_, sigma_a_, w);

        const Eigen::VectorXd r = y - h;

        // g = J^T R^{-1} r, with diagonal R^{-1} implemented as row-scaling
        for (int row = 0; row < J.rows(); ++row) {
            g.noalias() += (w(row) * r(row)) * J.row(row).transpose();
        }
    }

    return logLikelihood(x, system);
}

double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system,
                                         Eigen::VectorXd& g,
                                         Eigen::MatrixXd& H) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    // Assemble matched subsets
    std::vector<std::size_t> useL;
    std::vector<int> useF;
    const int N = static_cast<int>(A_.size());
    for (std::size_t j = 0; j < sys.numberLandmarks(); ++j) {
        if (j < idxFeatures_.size()) {
            const int f = idxFeatures_[j];
            if (f >= 0 && f < N) { useL.push_back(j); useF.push_back(f); }
        }
    }
    const std::size_t k = useL.size();

    g.setZero(x.size());
    H = Eigen::MatrixXd::Zero(x.size(), x.size());

    if (k > 0)
    {
        Eigen::VectorXd y(3 * k);
        for (std::size_t i = 0; i < k; ++i) {
            const int f = useF[i];
            y.segment<2>(3*i) = Yuv_.col(f);
            y(3*i + 2)        = A_(f);
        }

        Eigen::MatrixXd J;
        Eigen::VectorXd h = predictDuckBundle(x, J, sys, useL);

        Eigen::VectorXd w;
        buildInvWeights(k, sigma_c_, sigma_a_, w);
        const Eigen::VectorXd r = y - h;

        // Weighted rows (equivalent to J^T R^{-1} r and -J^T R^{-1} J)
        Eigen::MatrixXd JW = J;
        for (int row = 0; row < JW.rows(); ++row) JW.row(row) *= std::sqrt(w(row));
        Eigen::VectorXd rW = r;
        for (int i = 0; i < rW.size(); ++i) rW(i) *= std::sqrt(w(i));

        g.noalias() = JW.transpose() * rW;
        H.noalias() = -(JW.transpose() * JW);
    }

    return logLikelihood(x, system, g);
}
