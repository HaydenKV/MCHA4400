#pragma once
#include <vector>
#include <unordered_map>
#include "MeasurementSLAMPointBundle.h"

class MeasurementSLAMUniqueTagBundle : public MeasurementPointBundle
{
public:
    MeasurementSLAMUniqueTagBundle(double time,
                                   const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
                                   const Camera& camera,
                                   const std::vector<int>& ids);

    void setIdByLandmark(const std::vector<int>& m) { id_by_landmark_ = m; }
    const std::vector<int>& idByLandmark() const { return id_by_landmark_; }

    virtual const std::vector<int>& associate(const SystemSLAM& system,
                                              const std::vector<std::size_t>& idxLandmarks) override;
    
    virtual void update(SystemBase& system) override;

protected:
    Eigen::Matrix<double,8,1> predictTagCorners(const Eigen::VectorXd& x, 
                                                 const SystemSLAM& system, 
                                                 std::size_t idxLandmark) const;
    
    template<typename Scalar>
    Eigen::Matrix<Scalar,8,1> predictTagCornersT(const Eigen::VectorX<Scalar>& x,
                                                  const SystemSLAM& system,
                                                  std::size_t idxLandmark) const;

private:
    std::vector<int> ids_;
    std::vector<int> id_by_landmark_;
    
    static constexpr double TAG_SIZE = 0.166;
};