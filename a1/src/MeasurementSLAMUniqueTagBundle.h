#pragma once
#include <vector>
#include "MeasurementSLAMPointBundle.h"

/**
 * @brief Measurement class for ArUco tag SLAM with unique IDs
 * 
 * Handles:
 * - ID-based data association (trivial since tags have unique IDs)
 * - 4-corner measurement model per tag
 * - Grace period before deletion (10 frames while visible)
 * - Uncertainty inflation for visible-but-unassociated landmarks
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
    
    void setConsecutiveMisses(const std::vector<int>& m) { consecutive_misses_ = m; }
    const std::vector<int>& getConsecutiveMisses() const { return consecutive_misses_; }

    /**
     * @brief Check if landmark should be visualized as "associated"
     * Returns true if detected this frame OR within grace period
     */
    bool isEffectivelyAssociated(std::size_t landmarkIdx) const;

    /**
     * @brief ID-based data association
     * 
     * Matches detected tags with map landmarks using unique IDs.
     * Increments miss counter only for landmarks that are visible but not detected.
     */
    virtual const std::vector<int>& associate(const SystemSLAM& system,
                                              const std::vector<std::size_t>& idxLandmarks) override;
    
    /**
     * @brief Measurement update with uncertainty inflation and deletion
     * 
     * 1. Inflates uncertainty for visible-but-unassociated landmarks (red)
     * 2. Deletes landmarks after MAX_CONSECUTIVE_MISSES (10 frames)
     * 3. Performs Kalman update for associated landmarks
     */
    virtual void update(SystemBase& system) override;
    
    // Log-likelihood overrides for optimization
    virtual double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system) const override;
    virtual double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g) const override;
    virtual double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g, Eigen::MatrixXd& H) const override;

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
    std::vector<int> consecutive_misses_;   ///< Persistent: miss counter per landmark
    
    static constexpr double TAG_SIZE = 0.166;        ///< Tag edge length (meters)
    static constexpr int MAX_CONSECUTIVE_MISSES = 10; ///< Delete after this many misses
};