#include <cstddef>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "SystemSLAM.h"
#include "SystemSLAMPoseLandmarks.h"
#include "Camera.h"
#include "Measurement.h"
#include "MeasurementSLAM.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "rotation.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace {

    // Thread-local scratch for the current evaluation chain
    thread_local bool tl_assoc_ready = false;
    thread_local std::vector<std::size_t> tl_idxUseLandmarks;
    thread_local std::vector<int>         tl_idxUseFeatures;

    inline void ensureAssociatedOnce(const SystemSLAM& sys,
                                     const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
                                     const std::vector<int>& idxFeatures)
    {
        if (tl_assoc_ready) return;

        tl_idxUseLandmarks.clear();
        tl_idxUseFeatures.clear();
        tl_idxUseLandmarks.reserve(sys.numberLandmarks());
        tl_idxUseFeatures.reserve(sys.numberLandmarks());

        // CRITICAL FIX: Bounds check using N (number of tags), not Y.cols() (4N)
        const int N = static_cast<int>(Y.cols()) / 4;
        assert(Y.cols() % 4 == 0 && "Y must have 4 columns per tag");  // Guard

        const std::size_t nL = sys.numberLandmarks();
        for (std::size_t j = 0; j < nL; ++j)
        {
            if (j < idxFeatures.size()) {
                const int fi = idxFeatures[j];
                if (fi >= 0 && fi < N) {  // FIX: Use N, not Y.cols()
                    tl_idxUseLandmarks.push_back(j);
                    tl_idxUseFeatures.push_back(fi);
                }
            }
        }
        tl_assoc_ready = true;
    }

    inline void clearAssociationScratch() {
        tl_assoc_ready = false;
        tl_idxUseLandmarks.clear();
        tl_idxUseFeatures.clear();
    }
}

MeasurementSLAMUniqueTagBundle::MeasurementSLAMUniqueTagBundle(
    double time,
    const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
    const Camera& camera,
    const std::vector<int>& ids)
: MeasurementPointBundle(time, Y, camera)
, ids_(ids)
, id_by_landmark_()
{
    // GUARD: Y must have 4 columns per tag
    assert(Y_.cols() % 4 == 0 && 
           "Y packing error: must have 4 columns per tag detection");
    
    sigma_ = 2.0;
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM* MeasurementSLAMUniqueTagBundle::clone() const
{
    auto* copy = new MeasurementSLAMUniqueTagBundle(time_, Y_, camera_, ids_);
    copy->id_by_landmark_ = this->id_by_landmark_;
    copy->idxFeatures_ = this->idxFeatures_;
    copy->is_visible_ = this->is_visible_;
    copy->is_effectively_associated_ = this->is_effectively_associated_;
    copy->sigma_ = this->sigma_;
    return copy;
}

bool MeasurementSLAMUniqueTagBundle::isEffectivelyAssociated(std::size_t landmarkIdx) const
{
    if (landmarkIdx >= is_effectively_associated_.size()) {
        return false;
    }
    return is_effectively_associated_[landmarkIdx];
}

const std::vector<int>& MeasurementSLAMUniqueTagBundle::associate(
    const SystemSLAM& system,
    const std::vector<std::size_t>& /*idxLandmarks*/)
{
    const SystemSLAMPoseLandmarks& sysPose = dynamic_cast<const SystemSLAMPoseLandmarks&>(system);
    const std::size_t nL = sysPose.numberLandmarks();

    // Ensure persistent state vectors are sized correctly
    if (id_by_landmark_.size() < nL) {
        id_by_landmark_.resize(nL, -1);
    }

    // Reset per-frame state
    is_visible_.assign(nL, false);
    is_effectively_associated_.assign(nL, false);
    idxFeatures_.assign(nL, -1);

    // Build map: tag ID → feature index in measurement Y
    std::unordered_map<int, int> id2feat;
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        id2feat[ids_[i]] = static_cast<int>(i);
    }

    // Use PRIOR MEAN for visibility checks
    const Eigen::VectorXd xmean = sysPose.density.mean();

    for (std::size_t j = 0; j < nL; ++j)
    {
        const int tagId = id_by_landmark_[j];
        if (tagId < 0) continue;  // No tag ID assigned to this landmark

        // ====================================================================
        // STEP 1: ID-based Detection Association
        // ====================================================================
        // If a tag was detected (ID found in measurement), associate it.
        // Detection already passed all quality gates in detectArUcoPOSE,
        // so we trust the measurement regardless of our state estimate accuracy.
        
        auto it = id2feat.find(tagId);
        if (it != id2feat.end()) {
            const int featIdx = it->second;
            idxFeatures_[j] = featIdx;
            is_effectively_associated_[j] = true;
        }

        // ====================================================================
        // STEP 2: Predicted Visibility (for |U| penalty)
        // ====================================================================
        // Check if landmark SHOULD be visible based on current state estimate.
        // This is used for the |U| penalty term in log-likelihood.
        //
        // IMPORTANT: Must use xmean (prior mean), not the optimization variable!
        // This ensures the |U| penalty remains piecewise-constant during optimization.
        //
        // A landmark contributes to |U| if:
        // - It has a tag ID (initialized)
        // - Predicted to be visible (all 4 corners in FOV)
        // - NOT associated with a detection this frame
        
        Eigen::Matrix<double,8,1> corners = predictTagCorners(xmean, sysPose, j);
        
        // Conservative visibility check: ALL 4 corners must be safely inside image
        // Use areCornersInside() which includes a safety margin
        const bool allCornersVisible = camera_.areCornersInside(corners);
        is_visible_[j] = allCornersVisible;
        
        // Note: A landmark can be:
        // - is_visible_=true,  is_effectively_associated_=true  → Blue (detected & tracked)
        // - is_visible_=true,  is_effectively_associated_=false → Red (visible but missed)
        // - is_visible_=false, is_effectively_associated_=false → Yellow (out of FOV)
    }

    return idxFeatures_;
}

void MeasurementSLAMUniqueTagBundle::update(SystemBase& system)
{
    SystemSLAMPoseLandmarks& systemSLAM = dynamic_cast<SystemSLAMPoseLandmarks&>(system);

    // ID-based data association
    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);
    associate(systemSLAM, idxLandmarks);

    // SCENARIO 1: NO DELETION (tags persist for loop closure)
    
    // Perform measurement update
    Measurement::update(system);
}

Eigen::Matrix<double,8,1> MeasurementSLAMUniqueTagBundle::predictTagCorners(
    const Eigen::VectorXd& x,
    const SystemSLAM& system,
    std::size_t idxLandmark) const
{
    return predictTagCornersT<double>(x, system, idxLandmark);
}

template<typename Scalar>
Eigen::Matrix<Scalar,8,1> MeasurementSLAMUniqueTagBundle::predictTagCornersT(
    const Eigen::VectorX<Scalar>& x,
    const SystemSLAM& system,
    std::size_t idxLandmark) const
{
    // Get landmark state: [r^n_L/N (3), Θ^n_L (3)]
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rLNn = x.template segment<3>(idx);      // Position
    Eigen::Vector3<Scalar> ThetaLn = x.template segment<3>(idx+3); // Orientation (RPY)

    // Convert Euler angles to rotation matrix
    Eigen::Matrix3<Scalar> RnL = rpy2rot(ThetaLn);

    // Get BODY pose from state (NOT camera pose!)
    Pose<Scalar> Tnb;
    Tnb.translationVector = SystemSLAM::cameraPosition(camera_, x);  // r^n_B/N
    Tnb.rotationMatrix    = SystemSLAM::cameraOrientation(camera_, x); // R^n_b

    // T^n_c = T^n_b * T^b_c
    Pose<Scalar> Tnc = camera_.bodyToCamera(Tnb);
    
    // Camera orientation and position in world frame
    Eigen::Matrix3<Scalar> Rcn = Tnc.rotationMatrix.transpose(); // R^c_n = (R^n_c)^T
    Eigen::Vector3<Scalar> rCNn = Tnc.translationVector;         // r^n_C/N

    // Define 4 corners in tag's local frame (TL, TR, BR, BL)
    const Scalar half = TAG_SIZE / 2.0;
    Eigen::Matrix<Scalar,3,4> cornersL;
    cornersL.col(0) << -half,  half, Scalar(0);  // Top-Left
    cornersL.col(1) <<  half,  half, Scalar(0);  // Top-Right
    cornersL.col(2) <<  half, -half, Scalar(0);  // Bottom-Right
    cornersL.col(3) << -half, -half, Scalar(0);  // Bottom-Left

    // Project each corner to image
    Eigen::Matrix<Scalar,8,1> h; // Output: [u1,v1, u2,v2, u3,v3, u4,v4]^T

    for (int c = 0; c < 4; ++c)
    {
        // 1) Corner position in WORLD frame: r^n_{jc}/N = R^n_L * r^L_{jc}/L + r^n_L/N
        Eigen::Vector3<Scalar> rJcNn = rLNn + RnL * cornersL.col(c);

        // 2) Corner position in CAMERA frame: r^c_{jc}/C = R^c_n * (r^n_{jc}/N - r^n_C/N)
        Eigen::Vector3<Scalar> rJcCc = Rcn * (rJcNn - rCNn);

        // 3) Project to image using vectorToPixel
        Eigen::Vector2<Scalar> uv = camera_.vectorToPixel(rJcCc);

        h(2*c)     = uv(0);  // u
        h(2*c + 1) = uv(1);  // v
    }

    return h;
}

// ============================================================================
// LOG-LIKELIHOOD: PROPER BUILD-ON-TOP PATTERN (Following Lab8/9)
// ============================================================================

double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    // Build selection once per evaluation chain
    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    if (tl_idxUseLandmarks.empty()) {
        clearAssociationScratch();
        return 0.0;
    }

    const std::size_t k = tl_idxUseLandmarks.size();
    const double inv_R = 1.0 / (sigma_ * sigma_);

    double logL = 0.0;

    // Sum log-likelihoods for associated landmarks
    // Σ_{(i,j)∈A} Σ_{c=1}^{4} log p_A(y_{ic}|η, m_j)
    for (std::size_t i = 0; i < k; ++i)
    {
        const std::size_t j = tl_idxUseLandmarks[i];
        const int fi = tl_idxUseFeatures[i];

        // Measured corners y_i (8x1): [u1,v1, u2,v2, u3,v3, u4,v4]^T
        Eigen::Matrix<double,8,1> yi;
        for (int c = 0; c < 4; ++c)
            yi.segment<2>(2*c) = Y_.col(4*fi + c);

        // Predicted corners h_i(x)
        Eigen::Matrix<double,8,1> hi = predictTagCorners(x, sys, j);
        
        // Residual
        Eigen::Matrix<double,8,1> ri = yi - hi;

        // Add Gaussian log-likelihood (drop constants)
        // log N(y; h, σ²I) = -0.5 * (1/σ²) * ||r||²
        logL += -0.5 * inv_R * ri.squaredNorm();
    }

    // CRITICAL FIX: |U| penalty term - MUST use is_visible_ from associate()
    // which was computed using the PRIOR MEAN, not the optimization variable x!
    // 
    // The penalty is: -4|U| log|Y|
    // where |U| is the number of visible but unassociated landmarks
    // and |Y| is the image area in pixels.
    //
    // IMPORTANT: This term is piecewise-constant w.r.t. x, so it has:
    // - Zero gradient everywhere (except at discontinuities where it's undefined)
    // - Zero Hessian everywhere
    // Therefore, it is CORRECTLY omitted from gradient/Hessian computations.
    {
        const double imgArea =
            static_cast<double>(camera_.imageSize.width) *
            static_cast<double>(camera_.imageSize.height);
        
        int Ucount = 0;
        const std::size_t nL = sys.numberLandmarks();
        
        // Count visible but unassociated landmarks
        for (std::size_t j = 0; j < nL; ++j)
        {
            // Skip if associated this frame
            if (j < idxFeatures_.size() && idxFeatures_[j] >= 0) {
                continue;
            }
            
            // Skip if no tag ID assigned yet
            if (j >= id_by_landmark_.size() || id_by_landmark_[j] < 0) {
                continue;
            }
            
            // CRITICAL: Use is_visible_ flag set in associate() using PRIOR MEAN
            // NOT the optimization variable x!
            if (j < is_visible_.size() && is_visible_[j]) {
                ++Ucount;
            }
        }
        
        // Apply penalty: -4|U| log|Y|
        // Factor of 4 because each tag has 4 corners
        if (Ucount > 0 && imgArea > 0.0) {
            logL -= 4.0 * static_cast<double>(Ucount) * std::log(imgArea);
        }
    }

    clearAssociationScratch();
    return logL;
}

// GRADIENT VERSION: Adds gradient, then delegates to scalar
double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system,
    Eigen::VectorXd& g) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    // Build selection once per evaluation chain (no-op if already done)
    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    g.resize(x.size());
    g.setZero();

    if (!tl_idxUseLandmarks.empty())
    {
        const std::size_t k = tl_idxUseLandmarks.size();
        const double inv_R = 1.0 / (sigma_ * sigma_);

        for (std::size_t i = 0; i < k; ++i)
        {
            const std::size_t j = tl_idxUseLandmarks[i];
            const int fi = tl_idxUseFeatures[i];

            // Measured corners y_i (8x1)
            Eigen::Matrix<double,8,1> yi;
            for (int c = 0; c < 4; ++c)
                yi.segment<2>(2*c) = Y_.col(4*fi + c);

            // Autodiff for h_i and J_i
            using autodiff::dual;
            using autodiff::jacobian;
            using autodiff::wrt;
            using autodiff::at;
            using autodiff::val;

            Eigen::VectorX<dual> xdual = x.cast<dual>();
            auto hfun = [&](const Eigen::VectorX<dual>& xad)->Eigen::Matrix<dual,8,1> {
                return predictTagCornersT<dual>(xad, sys, j);
            };

            Eigen::Matrix<dual,8,1> hdual;
            Eigen::MatrixXd Ji = jacobian(hfun, wrt(xdual), at(xdual), hdual);

            Eigen::Matrix<double,8,1> hi;
            for (int t = 0; t < 8; ++t) hi(t) = val(hdual(t));

            const Eigen::Matrix<double,8,1> ri = yi - hi;

            // Gradient contribution: g += (1/σ²) * J^T * r
            g.noalias() += inv_R * (Ji.transpose() * ri);
        }
    }

    // NOTE: |U| penalty has zero gradient everywhere (piecewise-constant in x)
    // so it is correctly omitted from gradient computation.

    // Build-on style: return via scalar-only overload (will reuse selection and clear it)
    return logLikelihood(x, system);
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system,
    Eigen::VectorXd& g,
    Eigen::MatrixXd& H) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    // Build selection once per evaluation chain (no-op if already done)
    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    g.resize(x.size());       g.setZero();
    H.resize(x.size(), x.size());   H.setZero();

    if (!tl_idxUseLandmarks.empty())
    {
        const std::size_t k = tl_idxUseLandmarks.size();
        const double inv_R = 1.0 / (sigma_ * sigma_);

        for (std::size_t i = 0; i < k; ++i)
        {
            const std::size_t j = tl_idxUseLandmarks[i];
            const int fi = tl_idxUseFeatures[i];

            // Measured corners y_i (8x1)
            Eigen::Matrix<double,8,1> yi;
            for (int c = 0; c < 4; ++c)
                yi.segment<2>(2*c) = Y_.col(4*fi + c);

            // Autodiff for h_i and J_i
            using autodiff::dual;
            using autodiff::jacobian;
            using autodiff::wrt;
            using autodiff::at;
            using autodiff::val;

            Eigen::VectorX<dual> xdual = x.cast<dual>();
            auto hfun = [&](const Eigen::VectorX<dual>& xad)->Eigen::Matrix<dual,8,1> {
                return predictTagCornersT<dual>(xad, sys, j);
            };

            Eigen::Matrix<dual,8,1> hdual;
            Eigen::MatrixXd Ji = jacobian(hfun, wrt(xdual), at(xdual), hdual);

            Eigen::Matrix<double,8,1> hi;
            for (int t = 0; t < 8; ++t) hi(t) = val(hdual(t));

            const Eigen::Matrix<double,8,1> ri = yi - hi;

            // Gauss-Newton Hessian approximation: H += -(1/σ²) * J^T * J
            H.noalias() += -inv_R * (Ji.transpose() * Ji);
        }
    }

    // NOTE: |U| penalty has zero Hessian everywhere (piecewise-constant in x)
    // so it is correctly omitted from Hessian computation.

    // Build-on style: return via gradient overload (which calls scalar, reuses selection and clears)
    return logLikelihood(x, system, g);
}

// Explicit template instantiation
template Eigen::Matrix<double,8,1> MeasurementSLAMUniqueTagBundle::predictTagCornersT<double>(
    const Eigen::VectorXd&, const SystemSLAM&, std::size_t) const;
template Eigen::Matrix<autodiff::dual,8,1> MeasurementSLAMUniqueTagBundle::predictTagCornersT<autodiff::dual>(
    const Eigen::VectorX<autodiff::dual>&, const SystemSLAM&, std::size_t) const;