#ifndef SYSTEMSLAMPOSELANDMARKS_H
#define SYSTEMSLAMPOSELANDMARKS_H

#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"

/*
 * State containing body velocities, body pose and landmark poses
 *
 *     [ vBNb     ]  Body translational velocity (body-fixed)
 *     [ omegaBNb ]  Body angular velocity (body-fixed)
 *     [ rBNn     ]  Body position (world-fixed)
 *     [ Thetanb  ]  Body orientation (world-fixed)
 * x = [ rL1Nn     ]  Landmark 1 position (world-fixed)
 *     [ omegaL1Nc ]  Landmark 1 orientation (world-fixed)
 *     [ rL2Nn     ]  Landmark 2 position (world-fixed)
 *     [ omegaL2Nc ]  Landmark 2 orientation (world-fixed)
 *     [ ...       ]  ...
 *
 */
class SystemSLAMPoseLandmarks : public SystemSLAM
{
public:
    explicit SystemSLAMPoseLandmarks(const GaussianInfo<double> & density);
    SystemSLAM * clone() const override;
    virtual std::size_t numberLandmarks() const override;
    virtual std::size_t landmarkPositionIndex(std::size_t idxLandmark) const override;

    /// Append a pose-landmark (position r_n^j and Euler angles Theta_nj) with
    /// an initial square-root covariance Sj (6x6, upper-triangular).
    /// Returns the landmark index j just added.
    std::size_t appendLandmark(const Eigen::Vector3d& r_nL,
                               const Eigen::Vector3d& Theta_nL,
                               const Eigen::Matrix<double,6,6>& Sj);
};

#endif