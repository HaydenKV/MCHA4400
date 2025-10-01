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
{
    // Measurement noise: typical for corner detection is 1-2 pixels
    sigma_ = 1.0;
}

// ============================================================================
// DATA ASSOCIATION
// ============================================================================
const std::vector<int>& MeasurementSLAMUniqueTagBundle::associate(
    const SystemSLAM& system,
    const std::vector<std::size_t>& idxLandmarks)
{
    // Build map: tag ID → feature index (which detected tag has this ID)
    std::unordered_map<int,int> id2feat;
    id2feat.reserve(ids_.size());
    for (int i = 0; i < static_cast<int>(ids_.size()); ++i)
        id2feat.emplace(ids_[i], i);

    // Ensure mapping vector is sized correctly
    if (id_by_landmark_.size() < system.numberLandmarks())
        id_by_landmark_.resize(system.numberLandmarks(), -1);

    // Initialize association: -1 means unassociated
    idxFeatures_.assign(system.numberLandmarks(), -1);

    // For each landmark in the map, find its corresponding detected tag
    for (std::size_t j = 0; j < system.numberLandmarks(); ++j)
    {
        const int tagId = (j < id_by_landmark_.size()) ? id_by_landmark_[j] : -1;
        if (tagId < 0) continue;  // Landmark has no ID (shouldn't happen)

        // Look for this tag ID in detected features
        auto it = id2feat.find(tagId);
        if (it != id2feat.end())
            idxFeatures_[j] = it->second;  // Found! Associate landmark j with feature i
    }

    return idxFeatures_;
}

// ============================================================================
// PREDICTION: 4 CORNERS FOR A SINGLE TAG
// ============================================================================
template<typename Scalar>
Eigen::Matrix<Scalar,8,1> MeasurementSLAMUniqueTagBundle::predictTagCornersT(
    const Eigen::VectorX<Scalar>& x,
    const SystemSLAM& system,
    std::size_t idxLandmark) const
{
    // Get landmark state from x
    // State layout: [..., r^n_j/N (3), Θ^n_j (3), ...]
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rLNn = x.template segment<3>(idx);      // Landmark position
    Eigen::Vector3<Scalar> ThetaLn = x.template segment<3>(idx+3); // Landmark orientation (RPY)
    
    // Convert Euler angles to rotation matrix: R^n_L
    Eigen::Matrix3<Scalar> RnL = rpy2rot(ThetaLn);
    
    // Get camera pose from state
    Pose<Scalar> Tnb;
    Tnb.translationVector = SystemSLAM::cameraPosition(camera_, x);
    Tnb.rotationMatrix = SystemSLAM::cameraOrientation(camera_, x);
    
    // Define 4 corners in TAG's local frame (centered at tag origin)
    // Layout: TL, TR, BR, BL (matching OpenCV ArUco convention)
    const Scalar half = TAG_SIZE / 2.0;
    Eigen::Matrix<Scalar,3,4> cornersL;
    cornersL.col(0) << -half,  half, Scalar(0);  // Top-Left
    cornersL.col(1) <<  half,  half, Scalar(0);  // Top-Right
    cornersL.col(2) <<  half, -half, Scalar(0);  // Bottom-Right
    cornersL.col(3) << -half, -half, Scalar(0);  // Bottom-Left
    
    // Camera pose: R^c_n and r^n_C/N
    Eigen::Matrix3<Scalar> Rcn = Tnb.rotationMatrix.transpose();
    Eigen::Vector3<Scalar> rCNn = Tnb.translationVector;
    
    // Project each corner to image
    Eigen::Matrix<Scalar,8,1> h;  // Output: [u1,v1,u2,v2,u3,v3,u4,v4]^T
    
    for (int c = 0; c < 4; ++c)
    {
        // 1) Corner position in WORLD frame
        //    r^n_P/N = r^n_L/N + R^n_L * r^L_corner
        Eigen::Vector3<Scalar> rPNn = rLNn + RnL * cornersL.col(c);
        
        // 2) Corner position in CAMERA frame
        //    r^c_P/C = R^c_n * (r^n_P/N - r^n_C/N)
        Eigen::Vector3<Scalar> rPCc = Rcn * (rPNn - rCNn);
        
        // 3) Project to IMAGE coordinates
        //    [u; v] = vectorToPixel(r^c_P/C, θ_dist)
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

// ============================================================================
// LOG-LIKELIHOOD: HANDLE 4-CORNER STRUCTURE + PENALTY TERM
// ============================================================================

double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    
    // Must have association computed already (done in associate())
    assert(idxFeatures_.size() == sys.numberLandmarks() && 
           "Association must be called before logLikelihood");
    
    double logLik = 0.0;
    
    // Count associated landmarks
    size_t nAssociated = 0;
    
    // === TERM 1: Log-likelihood for ASSOCIATED landmarks ===
    // log p(y|x) = Σ Σ log p(y_ic | m_j)
    //            (i,j)∈A c=1..4
    
    const double inv_R = 1.0 / (sigma_ * sigma_);  // 1/σ²
    
    for (std::size_t j = 0; j < sys.numberLandmarks(); ++j)
    {
        const int featIdx = idxFeatures_[j];
        if (featIdx < 0) continue;  // Landmark j is NOT associated
        
        nAssociated++;
        
        // Get measured corners for this tag (4 corners = 8 values)
        Eigen::Matrix<double,8,1> y_corners;
        for (int c = 0; c < 4; ++c)
        {
            // Y_ layout: [tag0_c0, tag0_c1, tag0_c2, tag0_c3, tag1_c0, ...]
            y_corners.segment<2>(2*c) = Y_.col(4*featIdx + c);
        }
        
        // Predict corners for this landmark
        Eigen::Matrix<double,8,1> h_corners = predictTagCorners(x, sys, j);
        
        // Residual: measured - predicted
        Eigen::Matrix<double,8,1> r = y_corners - h_corners;
        
        // Log-likelihood (Gaussian): -½ rᵀ R⁻¹ r
        // Assuming independent corners with same σ
        logLik += -0.5 * inv_R * r.squaredNorm();
    }
    
    // === TERM 2: Penalty for UNASSOCIATED landmarks ===
    // -4|U| log |Y|
    //
    // This term penalizes having landmarks in the map that don't match
    // any detected tags. It's critical for preventing spurious landmarks.
    //
    // |U| = number of potentially visible landmarks that are unassociated
    // 4 = number of corners per tag
    // |Y| = image area in pixels
    
    const size_t nUnassociated = sys.numberLandmarks() - nAssociated;
    
    if (nUnassociated > 0)
    {
        const double imageArea = camera_.imageSize.width * camera_.imageSize.height;
        const double penalty = -4.0 * nUnassociated * std::log(imageArea);
        logLik += penalty;
    }
    
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
    
    size_t nAssociated = 0;
    const double inv_R = 1.0 / (sigma_ * sigma_);
    
    // === GRADIENT COMPUTATION USING AUTODIFF ===
    // For each associated landmark, compute ∂logL/∂x using forward-mode autodiff
    
    for (std::size_t j = 0; j < sys.numberLandmarks(); ++j)
    {
        const int featIdx = idxFeatures_[j];
        if (featIdx < 0) continue;
        
        nAssociated++;
        
        // Measured corners
        Eigen::Matrix<double,8,1> y_corners;
        for (int c = 0; c < 4; ++c)
            y_corners.segment<2>(2*c) = Y_.col(4*featIdx + c);
        
        // Use autodiff to compute gradient
        using autodiff::dual;
        using autodiff::jacobian;
        using autodiff::wrt;
        using autodiff::at;
        using autodiff::val;
        
        Eigen::VectorX<dual> xdual = x.cast<dual>();
        
        // Prediction function for this landmark
        auto h_func = [&](const Eigen::VectorX<dual>& xad) -> Eigen::Matrix<dual,8,1>
        {
            return predictTagCornersT<dual>(xad, sys, j);
        };
        
        // Compute Jacobian: J = ∂h/∂x (8 × nx)
        Eigen::Matrix<dual,8,1> h_dual;
        Eigen::MatrixXd J = jacobian(h_func, wrt(xdual), at(xdual), h_dual);
        
        // Convert to double
        Eigen::Matrix<double,8,1> h_corners;
        for (int i = 0; i < 8; ++i)
            h_corners(i) = val(h_dual(i));
        
        // Residual
        Eigen::Matrix<double,8,1> r = y_corners - h_corners;
        
        // Gradient: ∂logL/∂x = inv_R * Jᵀ * r
        g.noalias() += inv_R * (J.transpose() * r);
    }
    
    // Penalty term doesn't contribute to gradient (constant w.r.t. state)
    
    return logLikelihood(x, system);  // Reuse scalar version
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
    
    size_t nAssociated = 0;
    const double inv_R = 1.0 / (sigma_ * sigma_);
    
    // === GAUSS-NEWTON HESSIAN APPROXIMATION ===
    // H ≈ -inv_R * Jᵀ * J (ignoring second-order terms)
    
    for (std::size_t j = 0; j < sys.numberLandmarks(); ++j)
    {
        const int featIdx = idxFeatures_[j];
        if (featIdx < 0) continue;
        
        nAssociated++;
        
        // Measured corners
        Eigen::Matrix<double,8,1> y_corners;
        for (int c = 0; c < 4; ++c)
            y_corners.segment<2>(2*c) = Y_.col(4*featIdx + c);
        
        // Use autodiff to compute Jacobian
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
        H.noalias() += -inv_R * (J.transpose() * J);  // Gauss-Newton approximation
    }
    
    return logLikelihood(x, system);
}

// ============================================================================
// MEASUREMENT UPDATE
// ============================================================================
void MeasurementSLAMUniqueTagBundle::update(SystemBase& system)
{
    SystemSLAMPoseLandmarks& systemSLAM = dynamic_cast<SystemSLAMPoseLandmarks&>(system);

    // Build list of all landmarks
    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);

    // Perform data association (populates idxFeatures_)
    associate(systemSLAM, idxLandmarks);

    // Call base class measurement update (uses BFGS optimization)
    Measurement::update(system);
}