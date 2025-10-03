#include "MeasurementSLAMUniqueTagBundle.h"
#include "SystemSLAMPoseLandmarks.h"
#include "rotation.hpp"
#include <unordered_map>
#include <iostream>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace {
    // Thread-local scratch so value/grad/H share the exact same selection
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
, consecutive_misses_()
{
    // Measurement noise: ArUco corner detection is accurate to ~1 pixel
    sigma_ = 1.0;
}

MeasurementSLAM* MeasurementSLAMUniqueTagBundle::clone() const
{
    // CRITICAL: Must preserve derived type for dynamic_cast in Plot
    auto* copy = new MeasurementSLAMUniqueTagBundle(time_, Y_, camera_, ids_);
    
    // Copy persistent state
    copy->id_by_landmark_ = this->id_by_landmark_;
    copy->consecutive_misses_ = this->consecutive_misses_;
    copy->idxFeatures_ = this->idxFeatures_;
    copy->sigma_ = this->sigma_;
    
    return copy;
}

bool MeasurementSLAMUniqueTagBundle::isEffectivelyAssociated(std::size_t landmarkIdx) const
{
    // Blue only when the tag was actually detected & matched this frame.
    return (landmarkIdx < idxFeatures_.size() && idxFeatures_[landmarkIdx] >= 0);
}

const std::vector<int>& MeasurementSLAMUniqueTagBundle::associate(
    const SystemSLAM& system,
    const std::vector<std::size_t>& /*idxLandmarks*/)
{
    // Fast lookup: tag ID → feature index
    std::unordered_map<int,int> id2feat;
    id2feat.reserve(ids_.size());
    for (int i = 0; i < static_cast<int>(ids_.size()); ++i)
        id2feat.emplace(ids_[i], i);

    // Ensure persistent vectors match current #landmarks
    if (id_by_landmark_.size() < system.numberLandmarks())
        id_by_landmark_.resize(system.numberLandmarks(), -1);

    if (consecutive_misses_.size() < system.numberLandmarks())
        consecutive_misses_.resize(system.numberLandmarks(), 0);

    // Result (-1 = unassociated)
    idxFeatures_.assign(system.numberLandmarks(), -1);

    // Camera pose (mean) for visibility checks
    const Eigen::VectorXd x = system.density.mean();
    const Eigen::Vector3d rCNn = SystemSLAM::cameraPosition(camera_, x);
    const Eigen::Matrix3d Rnc  = SystemSLAM::cameraOrientation(camera_, x);

    const int W = camera_.imageSize.width;
    const int H = camera_.imageSize.height;
    const int BORDER_MARGIN = 12;

    auto inBoundsMargin = [&](double u, double v)->bool {
        return (u >= BORDER_MARGIN && u < (W-1-BORDER_MARGIN) &&
                v >= BORDER_MARGIN && v < (H-1-BORDER_MARGIN));
    };

    for (std::size_t j = 0; j < system.numberLandmarks(); ++j)
    {
        const int tagId = (j < id_by_landmark_.size()) ? id_by_landmark_[j] : -1;
        if (tagId < 0) continue; // should not happen

        auto it = id2feat.find(tagId);

        if (it != id2feat.end())
        {
            // Found this tag in detections — check border proximity before accepting
            const int featIdx = it->second;

            bool nearBorder = false;
            for (int c = 0; c < 4; ++c) {
                const double u = Y_(0, 4*featIdx + c);
                const double v = Y_(1, 4*featIdx + c);
                if (!inBoundsMargin(u, v)) { nearBorder = true; break; }
            }

            if (!nearBorder) {
                idxFeatures_[j] = featIdx;   // associate
                consecutive_misses_[j] = 0;  // reset
            } else {
                // Ignore this measurement (too close to edges) → do NOT increment misses
                // (we simply leave idxFeatures_[j] == -1)
            }
        }
        else
        {
            // Not detected — increment miss ONLY if it should be visible (strict FOV + margin)
            const size_t idxPos = system.landmarkPositionIndex(j);
            const Eigen::Vector3d rLNn = x.segment<3>(idxPos);
            const Eigen::Vector3d rLCc = Rnc.transpose() * (rLNn - rCNn);

            if (rLCc.z() > 0.1) {
                const cv::Vec2d pix = camera_.vectorToPixel(cv::Vec3d(rLCc(0), rLCc(1), rLCc(2)));
                if (std::isfinite(pix[0]) && std::isfinite(pix[1]) && inBoundsMargin(pix[0], pix[1])) {
                    consecutive_misses_[j]++;   // visible but missed
                }
                // else: outside FOV (or behind): do not increment
            }
        }
    }

    std::cout << "  [assoc] Detected " << ids_.size() << " tags, associated "
              << std::count_if(idxFeatures_.begin(), idxFeatures_.end(), [](int i){ return i >= 0; })
              << " landmarks\n";

    return idxFeatures_;
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
    Tnb.rotationMatrix = SystemSLAM::cameraOrientation(camera_, x);
    
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
        
        // 3) Project to IMAGE coordinates (handles distortion)
        Eigen::Vector2<Scalar> pixel = camera_.vectorToPixel(rPCc);
        
        // Store in output vector
        h.template segment<2>(2*c) = pixel;
    }
    
    return h;
}

Eigen::Matrix<double,8,1> MeasurementSLAMUniqueTagBundle::predictTagCorners(
    const Eigen::VectorXd& x,
    const SystemSLAM& system,
    std::size_t idxLandmark) const
{
    return predictTagCornersT<double>(x, system, idxLandmark);
}


double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    // Build selection once for this evaluation chain
    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    if (tl_idxUseLandmarks.empty()) {
        clearAssociationScratch();
        return 0.0;
    }

    const std::size_t k = tl_idxUseLandmarks.size();
    const double inv_R  = 1.0 / (sigma_ * sigma_);

    // Stack y = [u1 v1 u2 v2 ...] for the selected set (8 entries per tag)
    Eigen::VectorXd y(8 * k);
    for (std::size_t i = 0; i < k; ++i) {
        const int fi = tl_idxUseFeatures[i];
        for (int c = 0; c < 4; ++c)
            y.segment<2>(8*i + 2*c) = Y_.col(4*fi + c);
    }

    // Predict h for the same set
    Eigen::VectorXd h(8 * k);
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t j = tl_idxUseLandmarks[i];
        h.segment<8>(8*i) = predictTagCorners(x, sys, j);
    }

    const Eigen::VectorXd r = y - h;
    const double value = -0.5 * inv_R * r.squaredNorm();

    clearAssociationScratch();
    return value;
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

    if (!tl_idxUseLandmarks.empty())
    {
        const std::size_t k = tl_idxUseLandmarks.size();
        const double inv_R  = 1.0 / (sigma_ * sigma_);

        // Stack y and accumulate J^T r
        for (std::size_t i = 0; i < k; ++i)
        {
            const std::size_t j = tl_idxUseLandmarks[i];
            const int fi        = tl_idxUseFeatures[i];

            // y_i (8x1)
            Eigen::Matrix<double,8,1> yi;
            for (int c = 0; c < 4; ++c)
                yi.segment<2>(2*c) = Y_.col(4*fi + c);

            // Autodiff for h_i and J_i (8 x nx)
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

            // ∇ℓ += inv_R * Jᵀ r
            g.noalias() += inv_R * (Ji.transpose() * ri);
        }
    }

    // Return scalar value via the scalar-only overload (also clears scratch)
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

    g.resize(x.size()); g.setZero();
    H.resize(x.size(), x.size()); H.setZero();

    if (!tl_idxUseLandmarks.empty())
    {
        const std::size_t k = tl_idxUseLandmarks.size();
        const double inv_R  = 1.0 / (sigma_ * sigma_);

        for (std::size_t i = 0; i < k; ++i)
        {
            const std::size_t j = tl_idxUseLandmarks[i];
            const int fi        = tl_idxUseFeatures[i];

            // y_i (8x1)
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

            // Gauss–Newton ∇ℓ and ∇²ℓ
            g.noalias() += inv_R * (Ji.transpose() * ri);
            H.noalias() += -inv_R * (Ji.transpose() * Ji);
        }
    }

    // Reuse the (value,grad) overload so scratch is cleared once
    return logLikelihood(x, system, g);
}



void MeasurementSLAMUniqueTagBundle::update(SystemBase& system)
{
    SystemSLAMPoseLandmarks& systemSLAM = dynamic_cast<SystemSLAMPoseLandmarks&>(system);

    // 1) Associate (ID-based; strict FOV + border margin)
    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);
    associate(systemSLAM, idxLandmarks);

    // 3) update (this correctly shrinks/expands uncertainty via the math)
    Measurement::update(system);
}
