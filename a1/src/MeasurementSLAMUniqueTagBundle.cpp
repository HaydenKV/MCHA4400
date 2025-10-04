#include "MeasurementSLAMUniqueTagBundle.h"
#include "SystemSLAMPoseLandmarks.h"
#include "rotation.hpp"
#include "Pose.hpp"

#include <Eigen/Core>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <cstddef>
#include <cmath>
#include <iostream>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace {
    // Thread-local scratch for consistent value/grad/hessian evaluation
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

        const std::size_t nL = sys.numberLandmarks();
        for (std::size_t j = 0; j < nL; ++j)
        {
            if (j < idxFeatures.size()) {
                const int fi = idxFeatures[j];
                // Only keep truly associated features inside measurement bounds
                if (fi >= 0 && 4*fi + 3 < Y.cols()) {
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
    // Measurement noise: ArUco corner detection with subpixel refinement
    // Conservative estimate: 1.5 pixels standard deviation
    sigma_ = 3;
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM* MeasurementSLAMUniqueTagBundle::clone() const
{
    // CRITICAL: Must preserve derived type for dynamic_cast in Plot
    auto* copy = new MeasurementSLAMUniqueTagBundle(time_, Y_, camera_, ids_);

    // Copy persistent state
    copy->id_by_landmark_ = this->id_by_landmark_;
    copy->idxFeatures_ = this->idxFeatures_;
    copy->sigma_ = this->sigma_;

    return copy;
}

bool MeasurementSLAMUniqueTagBundle::isEffectivelyAssociated(std::size_t landmarkIdx) const
{
    // Check for persistent tag ID, not just current frame detection
    // A landmark is "associated" if it has a known tag ID in the map
    // This makes it stay BLUE even during detector dropout
    
    if (landmarkIdx >= id_by_landmark_.size()) {
        return false;  // Landmark not yet in persistent map
    }
    
    // Has a valid tag ID? → Part of the map → Blue
    // No tag ID? → Unidentified → Red (should never happen in Scenario 1)
    return id_by_landmark_[landmarkIdx] >= 0;
}

const std::vector<int>& MeasurementSLAMUniqueTagBundle::associate(
    const SystemSLAM& system,
    const std::vector<std::size_t>& /*idxLandmarks*/)
{
    const SystemSLAMPoseLandmarks& sysPose = dynamic_cast<const SystemSLAMPoseLandmarks&>(system);
    const std::size_t nL = sysPose.numberLandmarks();

    // Ensure id_by_landmark is sized correctly
    if (id_by_landmark_.size() < nL) {
        id_by_landmark_.resize(nL, -1);
    }

    // Initialize association to "no match"
    idxFeatures_.assign(nL, -1);

    // Build reverse map: detected tag ID → feature index
    std::unordered_map<int, int> id2feat;
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        id2feat[ids_[i]] = static_cast<int>(i);
    }

    // Get current camera pose for FOV checking
    const Eigen::VectorXd xmean = sysPose.density.mean();
    const int W = camera_.imageSize.width;
    const int H = camera_.imageSize.height;

    // ID-based association with conservative FOV checking
    for (std::size_t j = 0; j < nL; ++j)
    {
        const int tagId = id_by_landmark_[j];
        if (tagId < 0) continue;  // Uninitialized landmark

        // Check if this tag was detected
        auto it = id2feat.find(tagId);
        if (it == id2feat.end()) {
            // Tag not detected this frame
            continue;
        }

        const int featIdx = it->second;

        // Predict where corners should appear
        auto predictedCorners = predictTagCorners(xmean, sysPose, j);

        // Check if ALL corners are safely within image bounds (conservative margin)
        bool allCornersGood = true;
        for (int c = 0; c < 4; ++c) {
            const double u = predictedCorners(2*c);
            const double v = predictedCorners(2*c + 1);

            if (u < BORDER_MARGIN || u > (W - 1 - BORDER_MARGIN) ||
                v < BORDER_MARGIN || v > (H - 1 - BORDER_MARGIN)) {
                allCornersGood = false;
                break;
            }
        }

        if (allCornersGood) {
            // Associate this landmark with the detected feature
            idxFeatures_[j] = featIdx;
        }
    }

    return idxFeatures_;
}

void MeasurementSLAMUniqueTagBundle::update(SystemBase& system)
{
    SystemSLAMPoseLandmarks& systemSLAM = dynamic_cast<SystemSLAMPoseLandmarks&>(system);

    // 1) ID-based data association with conservative FOV checking
    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);
    associate(systemSLAM, idxLandmarks);

    // 2) SCENARIO 1: NO DELETION OF LANDMARKS
    //    Tags have unique IDs - we want loop closure to work!
    //    Landmarks persist even if not detected for many frames.

    // 3) Perform measurement update (Kalman update) for associated landmarks
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

    // Get camera pose from state
    Pose<Scalar> Tnb;
    Tnb.translationVector = SystemSLAM::cameraPosition(camera_, x);
    Tnb.rotationMatrix    = SystemSLAM::cameraOrientation(camera_, x);

    // Define 4 corners in tag's local frame (centered at tag origin)
    // Order: TL, TR, BR, BL (matches ArUco detection order)
    const Scalar half = TAG_SIZE / 2.0;
    Eigen::Matrix<Scalar,3,4> cornersL;
    cornersL.col(0) << -half,  half, Scalar(0);  // Top-Left
    cornersL.col(1) <<  half,  half, Scalar(0);  // Top-Right
    cornersL.col(2) <<  half, -half, Scalar(0);  // Bottom-Right
    cornersL.col(3) << -half, -half, Scalar(0);  // Bottom-Left

    // Camera orientation and position in world frame
    Eigen::Matrix3<Scalar> Rcn = Tnb.rotationMatrix.transpose(); // R^c_n
    Eigen::Vector3<Scalar> rCNn = Tnb.translationVector;         // r^n_C/N

    // Project each corner to image
    Eigen::Matrix<Scalar,8,1> h; // Output: [u1,v1, u2,v2, u3,v3, u4,v4]^T

    for (int c = 0; c < 4; ++c)
    {
        // 1) Corner position in WORLD frame
        //    r^n_P/N = r^n_L/N + R^n_L * r^L_corner
        Eigen::Vector3<Scalar> rPNn = rLNn + RnL * cornersL.col(c);

        // 2) Corner position in CAMERA frame
        //    r^c_P/C = R^c_n * (r^n_P/N - r^n_C/N)
        Eigen::Vector3<Scalar> rPCc = Rcn * (rPNn - rCNn);

        // 3) Project to image using vectorToPixel
        Eigen::Vector2<Scalar> uv = camera_.vectorToPixel(rPCc);

        h(2*c)     = uv(0);  // u
        h(2*c + 1) = uv(1);  // v
    }

    return h;
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    if (tl_idxUseLandmarks.empty()) {
        clearAssociationScratch();
        return 0.0;
    }

    const std::size_t k = tl_idxUseLandmarks.size();
    const double inv_R = 1.0 / (sigma_ * sigma_);

    double logL = 0.0;

    for (std::size_t i = 0; i < k; ++i)
    {
        const std::size_t j = tl_idxUseLandmarks[i];
        const int fi = tl_idxUseFeatures[i];

        // Measured corners y_i (8x1)
        Eigen::Matrix<double,8,1> yi;
        for (int c = 0; c < 4; ++c)
            yi.segment<2>(2*c) = Y_.col(4*fi + c);

        // Predicted corners h_i (8x1)
        Eigen::Matrix<double,8,1> hi = predictTagCorners(x, sys, j);

        // Residual
        Eigen::Matrix<double,8,1> ri = yi - hi;

        // Log-likelihood contribution (Gaussian)
        logL += -0.5 * inv_R * ri.squaredNorm();
    }

    // ---- Assignment penalty: visible but unassociated tags ----
    // |Y| = image area (pixels); U = set of visible landmarks without a detection this frame.
    // Visible := all 4 predicted corners inside image bounds (conservative margin).
    {
        const int W = camera_.imageSize.width;
        const int H = camera_.imageSize.height;
        const double imgArea = static_cast<double>(W) * static_cast<double>(H);
        int Ucount = 0;
        const std::size_t nL = sys.numberLandmarks();
        for (std::size_t j = 0; j < nL; ++j)
        {
            // only consider landmarks with known IDs and not associated this frame
            if (j < idxFeatures_.size() && idxFeatures_[j] < 0 && id_by_landmark_.size() > j && id_by_landmark_[j] >= 0)
            {
                Eigen::Matrix<double,8,1> pj = predictTagCorners(x, sys, j);
                bool inView = true;
                for (int c = 0; c < 4; ++c) {
                    const double u = pj(2*c), v = pj(2*c+1);
                    if (u < BORDER_MARGIN || u > (W - 1 - BORDER_MARGIN) ||
                        v < BORDER_MARGIN || v > (H - 1 - BORDER_MARGIN)) { inView = false; break; }
                }
                if (inView) ++Ucount;
            }
        }
        if (Ucount > 0 && imgArea > 0.0) {
            logL += -4.0 * static_cast<double>(Ucount) * std::log(imgArea);
        }
    }

    clearAssociationScratch();
    return logL;
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system,
    Eigen::VectorXd& g) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    g.resize(x.size());
    g.setZero();

    if (tl_idxUseLandmarks.empty()) {
        // No associated pairs: just return scalar value (will clear scratch).
        return logLikelihood(x, system);
    }

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

        // Gradient: ∇ℓ = J^T * R^(-1) * r
        g.noalias() += inv_R * (Ji.transpose() * ri);
    }

    // Return scalar value via scalar-only overload (clears scratch and includes penalty term).
    return logLikelihood(x, system);
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system,
    Eigen::VectorXd& g,
    Eigen::MatrixXd& H) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    g.resize(x.size());   g.setZero();
    H.resize(x.size(), x.size());     H.setZero();

    if (tl_idxUseLandmarks.empty()) {
        // No associations: skip derivatives; return scalar (clears scratch, adds penalty).
        return logLikelihood(x, system);
    }

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

        // Gauss–Newton gradient and Hessian
        g.noalias() += inv_R * (Ji.transpose() * ri);
        H.noalias() += -inv_R * (Ji.transpose() * Ji);
    }

    // Finish via scalar-only overload to include penalty once and clear scratch.
    return logLikelihood(x, system);
}
