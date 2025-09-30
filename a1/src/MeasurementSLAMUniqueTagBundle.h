#pragma once
#include <vector>
#include <unordered_map>
#include "MeasurementSLAMPointBundle.h"

class MeasurementSLAMUniqueTagBundle : public MeasurementPointBundle
{
public:
    // Yc is 2xN (centroid per detected tag), ids is length N (OpenCV order)
    MeasurementSLAMUniqueTagBundle(double time,
                                   const Eigen::Matrix<double,2,Eigen::Dynamic>& Yc,
                                   const Camera& camera,
                                   const std::vector<int>& ids)
    : MeasurementPointBundle(time, Yc, camera)
    , ids_(ids)
    , id_by_landmark_()
    {
        // scenario 1: pixel noise ~ 2 px is reasonable
        this->sigma_ = 2.0;
    }

    // Set persistent mapping (landmark index -> tag ID)
    void setIdByLandmark(const std::vector<int>& m) { id_by_landmark_ = m; }

    // Accessor (optional)
    const std::vector<int>& idByLandmark() const { return id_by_landmark_; }

    // Override: associate landmarks to current frame features by ID
    const std::vector<int>& associate(const SystemSLAM& system,
                                      const std::vector<std::size_t>& idxLandmarks) override;

private:
    std::vector<int> ids_;            // current frame tag IDs (aligned with columns of Y_)
    std::vector<int> id_by_landmark_; // persistent mapping: landmark j -> tag ID
};
