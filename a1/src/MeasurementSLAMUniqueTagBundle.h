#pragma once
#include <vector>
#include <unordered_map>
#include "MeasurementSLAMPointBundle.h"

/**
 * @brief Measurement class for ArUco tag SLAM with unique IDs
 * 
 * This class handles:
 * 1. ID-based data association (trivial since tags have unique IDs)
 * 2. 4-corner measurement model per tag
 * 3. Log-likelihood with penalty term for unassociated landmarks
 * 
 * State: Each landmark is 6-DOF pose [r^n_j/N; Θ^n_j]
 * Measurement: 4 corners per tag [u1,v1,u2,v2,u3,v3,u4,v4]^T
 */
class MeasurementSLAMUniqueTagBundle : public MeasurementPointBundle
{
public:
    /**
     * @param time Measurement timestamp
     * @param Y Measurement matrix (2 × 4N) - all corners of all tags
     * @param camera Camera calibration
     * @param ids Tag IDs corresponding to measurements
     */
    MeasurementSLAMUniqueTagBundle(double time,
                                   const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
                                   const Camera& camera,
                                   const std::vector<int>& ids);

    void setIdByLandmark(const std::vector<int>& m) { id_by_landmark_ = m; }
    const std::vector<int>& idByLandmark() const { return id_by_landmark_; }

    /**
     * @brief ID-based data association
     * 
     * For each landmark, find the detected tag with matching ID.
     * Returns idxFeatures_[j] = feature index for landmark j (-1 if unassociated)
     */
    virtual const std::vector<int>& associate(const SystemSLAM& system,
                                              const std::vector<std::size_t>& idxLandmarks) override;
    
    virtual void update(SystemBase& system) override;

    // OVERRIDE LOG-LIKELIHOOD TO HANDLE 4-CORNER STRUCTURE AND PENALTY TERM
    virtual double logLikelihood(const Eigen::VectorXd& x, 
                                 const SystemEstimator& system) const override;
    
    virtual double logLikelihood(const Eigen::VectorXd& x, 
                                 const SystemEstimator& system,
                                 Eigen::VectorXd& g) const override;
    
    virtual double logLikelihood(const Eigen::VectorXd& x, 
                                 const SystemEstimator& system,
                                 Eigen::VectorXd& g, 
                                 Eigen::MatrixXd& H) const override;

    const std::vector<int>& idxFeatures() const { return idxFeatures_; }

protected:
    /**
     * @brief Predict 4 corner pixel locations for a single tag landmark
     * 
     * Given state x containing landmark [r^n_j/N; Θ^n_j], predicts where
     * the 4 corners of the tag should appear in the image.
     * 
     * @return 8×1 vector [u1,v1,u2,v2,u3,v3,u4,v4]^T
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
    std::vector<int> ids_;              ///< Tag IDs from detection
    std::vector<int> id_by_landmark_;   ///< Mapping: landmark index → tag ID
    
    static constexpr double TAG_SIZE = 0.166;  ///< Tag edge length in meters
};