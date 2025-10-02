#include "MeasurementSLAMUniqueTagBundle.h"
#include "SystemSLAMPoseLandmarks.h"
#include "rotation.hpp"
#include <unordered_map>
#include <iostream>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

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
    // Directly associated this frame?
    if (landmarkIdx < idxFeatures_.size() && idxFeatures_[landmarkIdx] >= 0) {
        return true;
    }
    
    // Within grace period (missed but not yet deleted)?
    if (landmarkIdx < consecutive_misses_.size()) {
        int missCount = consecutive_misses_[landmarkIdx];
        if (missCount > 0 && missCount <= MAX_CONSECUTIVE_MISSES) {
            return true;  // In grace period - show as blue
        }
    }
    
    return false;
}

const std::vector<int>& MeasurementSLAMUniqueTagBundle::associate(
    const SystemSLAM& system,
    const std::vector<std::size_t>& idxLandmarks)
{
    // Build fast lookup: tag ID → feature index
    std::unordered_map<int,int> id2feat;
    id2feat.reserve(ids_.size());
    for (int i = 0; i < static_cast<int>(ids_.size()); ++i)
        id2feat.emplace(ids_[i], i);

    // Ensure persistent vectors are sized to match current landmarks
    if (id_by_landmark_.size() < system.numberLandmarks())
        id_by_landmark_.resize(system.numberLandmarks(), -1);
    
    if (consecutive_misses_.size() < system.numberLandmarks())
        consecutive_misses_.resize(system.numberLandmarks(), 0);

    // Initialize association result (-1 = unassociated)
    idxFeatures_.assign(system.numberLandmarks(), -1);
    
    // Get camera pose for visibility checks
    const Eigen::VectorXd x = system.density.mean();
    const Eigen::Vector3d rCNn = SystemSLAM::cameraPosition(camera_, x);
    const Eigen::Matrix3d Rnc = SystemSLAM::cameraOrientation(camera_, x);

    // For each landmark in the map
    for (std::size_t j = 0; j < system.numberLandmarks(); ++j)
    {
        const int tagId = (j < id_by_landmark_.size()) ? id_by_landmark_[j] : -1;
        if (tagId < 0) continue; // No ID assigned (shouldn't happen)

        // Look for this tag ID in current detections
        auto it = id2feat.find(tagId);
        
        if (it != id2feat.end()) {
            // ASSOCIATED: Tag detected this frame
            idxFeatures_[j] = it->second;
            consecutive_misses_[j] = 0; // Reset miss counter
            
        } else {
            // NOT DETECTED: Check if it SHOULD be visible
            
            // Get landmark position from state
            size_t idx = system.landmarkPositionIndex(j);
            Eigen::Vector3d rLNn = x.segment<3>(idx);
            
            // Transform to camera frame
            Eigen::Vector3d rLCc = Rnc.transpose() * (rLNn - rCNn);
            double distance = rLCc.norm();
            
            // Check if potentially visible:
            // - In front of camera (z > 0.1m)
            // - Not too close (distance > 0.15m, ArUco fails when tag fills frame)
            // - Not too far (distance < 6m, detection unreliable beyond this)
            // - Within camera FOV
            if (rLCc.z() > 0.1 && distance > 0.15 && distance < 6.0) {
                // Convert to OpenCV format for FOV check
                cv::Vec3d rLCc_cv(rLCc(0), rLCc(1), rLCc(2));
                
                if (camera_.isVectorWithinFOV(rLCc_cv)) {
                    // Potentially visible but not detected → increment miss counter
                    consecutive_misses_[j]++;
                    // Note: Don't print here, too verbose
                }
                // else: outside FOV → don't increment (preserve in map)
            }
            // else: behind camera or too close/far → don't increment (preserve in map)
        }
    }
    std::cout << "  [assoc] Detected " << ids_.size() << " tags, associated " 
          << std::count_if(idxFeatures_.begin(), idxFeatures_.end(), 
                          [](int i){ return i >= 0; })
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
    
    double logLik = 0.0;
    size_t nTrulyUnassociated = 0;
    int nUsed = 0;
    const double inv_R = 1.0 / (sigma_ * sigma_);
    
    // Sum log-likelihood over all landmarks
    for (std::size_t j = 0; j < sys.numberLandmarks(); ++j)
    {
        const int featIdx = idxFeatures_[j];
        
        if (featIdx >= 0) {
            // ASSOCIATED: Compute Gaussian likelihood for 4 corners
            nUsed++;
            // Get measured corners (2×4 = 8 values)
            Eigen::Matrix<double,8,1> y_corners;
            for (int c = 0; c < 4; ++c)
                y_corners.segment<2>(2*c) = Y_.col(4*featIdx + c);
            
            // Predict corners from current state
            Eigen::Matrix<double,8,1> h_corners = predictTagCorners(x, sys, j);
            
            // Residual
            Eigen::Matrix<double,8,1> r = y_corners - h_corners;
            
            // Log-likelihood: -½(y-h)ᵀR⁻¹(y-h)
            // Assuming independent corners with same σ
            logLik += -0.5 * inv_R * r.squaredNorm();
            
        } else {
            // UNASSOCIATED: Check if truly lost (beyond grace period)
            int missCount = (j < consecutive_misses_.size()) ? consecutive_misses_[j] : 0;
            if (missCount > MAX_CONSECUTIVE_MISSES) {
                nTrulyUnassociated++;
            }
            // else: within grace period, no penalty
        }
    }
    
    // Penalty for truly unassociated landmarks (course requirement)
    if (nTrulyUnassociated > 0) {
        const double imageArea = camera_.imageSize.width * camera_.imageSize.height;
        const double penalty = -4.0 * nTrulyUnassociated * std::log(imageArea);
        logLik += penalty;
    }
    std::cout << "  [logLik] Using " << nUsed << "/" << sys.numberLandmarks() 
              << " landmarks, logLik=" << logLik << "\n";  // ADD THIS
                  
    return logLik;
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system,
    Eigen::VectorXd& g) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    
    g.resize(x.size());
    g.setZero();
    
    const double inv_R = 1.0 / (sigma_ * sigma_);
    
    // Compute gradient using autodiff
    for (std::size_t j = 0; j < sys.numberLandmarks(); ++j)
    {
        const int featIdx = idxFeatures_[j];
        if (featIdx < 0) continue; // Skip unassociated
        
        // Get measured corners
        Eigen::Matrix<double,8,1> y_corners;
        for (int c = 0; c < 4; ++c)
            y_corners.segment<2>(2*c) = Y_.col(4*featIdx + c);
        
        // Autodiff setup
        using autodiff::dual;
        using autodiff::jacobian;
        using autodiff::wrt;
        using autodiff::at;
        using autodiff::val;
        
        Eigen::VectorX<dual> xdual = x.cast<dual>();
        
        // Prediction function
        auto h_func = [&](const Eigen::VectorX<dual>& xad) -> Eigen::Matrix<dual,8,1>
        {
            return predictTagCornersT<dual>(xad, sys, j);
        };
        
        // Compute Jacobian: J = ∂h/∂x (8 × nx)
        Eigen::Matrix<dual,8,1> h_dual;
        Eigen::MatrixXd J = jacobian(h_func, wrt(xdual), at(xdual), h_dual);
        
        // Extract double values
        Eigen::Matrix<double,8,1> h_corners;
        for (int i = 0; i < 8; ++i)
            h_corners(i) = val(h_dual(i));
        
        // Residual
        Eigen::Matrix<double,8,1> r = y_corners - h_corners;
        
        // Gradient: ∂logL/∂x = inv_R * Jᵀ * r
        g.noalias() += inv_R * (J.transpose() * r);
    }
    
    // Penalty term doesn't contribute to gradient (constant w.r.t. state)
    
    return logLikelihood(x, system);
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system,
    Eigen::VectorXd& g,
    Eigen::MatrixXd& H) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    
    g.resize(x.size());
    g.setZero();
    H.resize(x.size(), x.size());
    H.setZero();
    
    const double inv_R = 1.0 / (sigma_ * sigma_);
    
    // Compute gradient and Gauss-Newton Hessian
    for (std::size_t j = 0; j < sys.numberLandmarks(); ++j)
    {
        const int featIdx = idxFeatures_[j];
        if (featIdx < 0) continue;
        
        Eigen::Matrix<double,8,1> y_corners;
        for (int c = 0; c < 4; ++c)
            y_corners.segment<2>(2*c) = Y_.col(4*featIdx + c);
        
        using autodiff::dual;
        using autodiff::jacobian;
        using autodiff::wrt;
        using autodiff::at;
        using autodiff::val;
        
        Eigen::VectorX<dual> xdual = x.cast<dual>();
        
        auto h_func = [&](const Eigen::VectorX<dual>& xad) -> Eigen::Matrix<dual,8,1>
        {
            return predictTagCornersT<dual>(xad, sys, j);
        };
        
        Eigen::Matrix<dual,8,1> h_dual;
        Eigen::MatrixXd J = jacobian(h_func, wrt(xdual), at(xdual), h_dual);
        
        Eigen::Matrix<double,8,1> h_corners;
        for (int i = 0; i < 8; ++i)
            h_corners(i) = val(h_dual(i));
        
        Eigen::Matrix<double,8,1> r = y_corners - h_corners;
        
        // Accumulate gradient and Hessian
        g.noalias() += inv_R * (J.transpose() * r);
        H.noalias() += -inv_R * (J.transpose() * J); // Gauss-Newton approximation
    }
    
    return logLikelihood(x, system);
}

void MeasurementSLAMUniqueTagBundle::update(SystemBase& system)
{
    SystemSLAMPoseLandmarks& systemSLAM = dynamic_cast<SystemSLAMPoseLandmarks&>(system);

    // 1. Associate
    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);
    associate(systemSLAM, idxLandmarks);

    // 2. Delete after consecutive misses
    std::vector<std::size_t> landmarksToRemove;
    for (size_t j = 0; j < systemSLAM.numberLandmarks(); ++j) {
        if (j < consecutive_misses_.size() && consecutive_misses_[j] > MAX_CONSECUTIVE_MISSES) {
            landmarksToRemove.push_back(j);
        }
    }
    
    for (auto it = landmarksToRemove.rbegin(); it != landmarksToRemove.rend(); ++it) {
        size_t j = *it;
        std::vector<Eigen::Index> keepIndices;
        size_t lmStartIdx = systemSLAM.landmarkPositionIndex(j);
        
        for (Eigen::Index i = 0; i < systemSLAM.density.dim(); ++i) {
            if (i < lmStartIdx || i >= lmStartIdx + 6) {
                keepIndices.push_back(i);
            }
        }
        
        systemSLAM.density = systemSLAM.density.marginal(keepIndices);
        id_by_landmark_.erase(id_by_landmark_.begin() + j);
        consecutive_misses_.erase(consecutive_misses_.begin() + j);
        idxFeatures_.erase(idxFeatures_.begin() + j);
    }

    // 3. Measurement update
    // The coupled EKF handles uncertainty correctly:
    // - Associated landmarks: uncertainty decreases
    // - Unassociated landmarks: uncertainty unchanged (measurement update skipped)
    // - All landmarks: tiny uncertainty growth from camera motion (via time update correlations)
    Measurement::update(system);
    
}