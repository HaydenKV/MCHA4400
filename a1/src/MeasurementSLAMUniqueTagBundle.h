#pragma once
#include <vector>
#include "MeasurementSLAMPointBundle.h"

/**
 * @brief Measurement class for ArUco tag SLAM with unique IDs (Scenario 1)
 * 
 * Key features for Scenario 1:
 * - ID-based data association (trivial since tags have unique IDs)
 * - 4-corner measurement model per tag
 * - NO DELETION of landmarks (tags persist for loop closure)
 * - Visual distinction: blue=associated, red=unassociated
 * 
 * IMPORTANT: Left pane ellipses show TAG CENTER (not corners!)
 * - Base class predictFeature() reads only position part of pose landmark
 * - For pose landmarks: state = [rLNn(3), ThetaLn(3)]
 * - predictFeature() reads only first 3 elements → tag center position
 */
class MeasurementSLAMUniqueTagBundle : public MeasurementPointBundle
{
public:
    /**
     * @param time Measurement timestamp
     * @param Y Measurement matrix (2 × 4N) - all 4 corners of all detected tags
     * @param camera Camera calibration
     * @param ids Tag IDs corresponding to measurements
     */
    MeasurementSLAMUniqueTagBundle(double time,
                                   const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
                                   const Camera& camera,
                                   const std::vector<int>& ids);

    /**
     * @brief Clone this measurement (must preserve derived type for Plot)
     */
    MeasurementSLAM* clone() const override;
    
    // Persistent state management (survives across frames)
    void setIdByLandmark(const std::vector<int>& m) { id_by_landmark_ = m; }
    const std::vector<int>& idByLandmark() const { return id_by_landmark_; }

    /**
     * @brief Check if landmark was effectively associated this frame
     * Used by Plot to color ellipses (blue=associated, red=unassociated)
     */
    bool isEffectivelyAssociated(std::size_t landmarkIdx) const;
    
    /**
     * @brief ID-based data association
     * 
     * Matches detected tags with map landmarks using unique IDs.
     * Conservative FOV checking to reject measurements near image borders.
     */
    virtual const std::vector<int>& associate(const SystemSLAM& system,
                                              const std::vector<std::size_t>& idxLandmarks) override;
    
    /**
     * @brief Measurement update WITHOUT deletion (Scenario 1)
     * 
     * Performs Kalman update for associated landmarks.
     * DOES NOT delete landmarks - tags persist for loop closure.
     */
    virtual void update(SystemBase& system) override;
    
    // Log-likelihood overrides for optimization
    virtual double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system) const override;
    virtual double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g) const override;
    virtual double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g, Eigen::MatrixXd& H) const override;

    const std::vector<bool>& isVisible() const { return is_visible_; }

protected:
    /**
     * @brief Predict 4 corner pixel locations for a tag landmark
     * @return 8×1 vector [u1,v1, u2,v2, u3,v3, u4,v4]^T
     */
    Eigen::Matrix<double,8,1> predictTagCorners(const Eigen::VectorXd& x, 
                                                 const SystemSLAM& system, 
                                                 std::size_t idxLandmark) const;
    
    /**
     * @brief Templated version for autodiff
     */
    template<typename Scalar>
    Eigen::Matrix<Scalar,8,1> predictTagCornersT(const Eigen::VectorX<Scalar>& x,
                                                  const SystemSLAM& system,
                                                  std::size_t idxLandmark) const;

private:
    std::vector<int> ids_;                  ///< Detected tag IDs this frame
    std::vector<int> id_by_landmark_;       ///< Persistent: landmark index → tag ID
    std::vector<bool> is_visible_;              // In FOV (geometric check)
    std::vector<bool> is_effectively_associated_; // In FOV + passed BORDER_MARGIN + detected
    
    static constexpr double TAG_SIZE = 0.166;        ///< Tag edge length (meters)
    static constexpr int BORDER_MARGIN = 15;         ///< Conservative FOV margin (pixels) ← FIXED: Now consistent!
};