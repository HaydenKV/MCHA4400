#include "MeasurementSLAMUniqueTagBundle.h"
#include "SystemSLAMPoseLandmarks.h"
#include "rotation.hpp"
#include <unordered_map>
#include <iostream>

MeasurementSLAMUniqueTagBundle::MeasurementSLAMUniqueTagBundle(
    double time,
    const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
    const Camera& camera,
    const std::vector<int>& ids)
: MeasurementPointBundle(time, Y, camera)
, ids_(ids)
, id_by_landmark_()
{
    sigma_ = 1.0;
}

const std::vector<int>& MeasurementSLAMUniqueTagBundle::associate(
    const SystemSLAM& system,
    const std::vector<std::size_t>& idxLandmarks)
{
    std::unordered_map<int,int> id2feat;
    id2feat.reserve(ids_.size());
    for (int i = 0; i < static_cast<int>(ids_.size()); ++i)
        id2feat.emplace(ids_[i], i);

    if (id_by_landmark_.size() < system.numberLandmarks())
        id_by_landmark_.resize(system.numberLandmarks(), -1);

    idxFeatures_.assign(system.numberLandmarks(), -1);

    for (std::size_t j = 0; j < system.numberLandmarks(); ++j)
    {
        const int tagId = (j < id_by_landmark_.size()) ? id_by_landmark_[j] : -1;
        if (tagId < 0) continue;

        auto it = id2feat.find(tagId);
        if (it != id2feat.end())
            idxFeatures_[j] = it->second;
    }

    return idxFeatures_;
}

template<typename Scalar>
Eigen::Matrix<Scalar,8,1> MeasurementSLAMUniqueTagBundle::predictTagCornersT(
    const Eigen::VectorX<Scalar>& x,
    const SystemSLAM& system,
    std::size_t idxLandmark) const
{
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rLNn = x.template segment<3>(idx);
    Eigen::Vector3<Scalar> ThetaLn = x.template segment<3>(idx+3);
    
    Eigen::Matrix3<Scalar> RnL = rpy2rot(ThetaLn);
    
    Pose<Scalar> Tnb;
    Tnb.translationVector = SystemSLAM::cameraPosition(camera_, x);
    Tnb.rotationMatrix = SystemSLAM::cameraOrientation(camera_, x);
    
    const Scalar half = TAG_SIZE / 2.0;
    Eigen::Matrix<Scalar,3,4> cornersL;
    cornersL.col(0) << -half,  half, 0;
    cornersL.col(1) <<  half,  half, 0;
    cornersL.col(2) <<  half, -half, 0;
    cornersL.col(3) << -half, -half, 0;
    
    Eigen::Matrix3<Scalar> Rcn = Tnb.rotationMatrix.transpose();
    Eigen::Vector3<Scalar> rCNn = Tnb.translationVector;
    
    Eigen::Matrix<Scalar,8,1> h;
    for (int c = 0; c < 4; ++c)
    {
        Eigen::Vector3<Scalar> rPNn = rLNn + RnL * cornersL.col(c);
        Eigen::Vector3<Scalar> rPCc = Rcn * (rPNn - rCNn);
        Eigen::Vector2<Scalar> pixel = camera_.vectorToPixel(rPCc);
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

void MeasurementSLAMUniqueTagBundle::update(SystemBase& system)
{
    SystemSLAMPoseLandmarks& systemSLAM = dynamic_cast<SystemSLAMPoseLandmarks&>(system);

    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);

    associate(systemSLAM, idxLandmarks);

    Measurement::update(system);
}