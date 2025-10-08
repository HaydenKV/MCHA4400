#ifndef MEASUREMENTSLAMDUCKBUNDLE_H
#define MEASUREMENTSLAMDUCKBUNDLE_H

#include <Eigen/Core>
#include "MeasurementSLAMPointBundle.h"
#include "SystemSLAM.h"
#include "Camera.h"

/*
 * MeasurementSLAMDuckBundle:
 *  - Inherits MeasurementPointBundle to reuse 2D feature projection & SNN association.
 *  - Extends the measurement vector with an area term per feature: [u, v, A]^T.
 *  - Likelihood uses diagonal noise (σ_px for u,v and σ_A for area).
 *
 * Usage: construct with (time, Yuv (2 x M), Avec (M), camera, duck_radius_m, sigma_px, sigma_area)
 */
class MeasurementSLAMDuckBundle : public MeasurementPointBundle
{
public:
    MeasurementSLAMDuckBundle(double time,
                              const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                              const Eigen::VectorXd& Avec,
                              const Camera& camera,
                              double duck_radius_m,
                              double sigma_px = 1.0,
                              double sigma_area = 150.0);

    MeasurementSLAM* clone() const override;

    // Value / gradient / Hessian of log-likelihood with (u,v,A) per associated landmark
    double logLikelihood(const Eigen::VectorXd& x,
                         const SystemEstimator& system) const override;

    double logLikelihood(const Eigen::VectorXd& x,
                         const SystemEstimator& system,
                         Eigen::VectorXd& g) const override;

    double logLikelihood(const Eigen::VectorXd& x,
                         const SystemEstimator& system,
                         Eigen::VectorXd& g,
                         Eigen::MatrixXd& H) const override;

protected:
    void update(SystemBase& system) override;

    // ---- Single landmark prediction: [u, v, A]^T ----
    template <typename Scalar>
    Eigen::Matrix<Scalar,3,1> predictDuckFeature(const Eigen::Matrix<Scalar,-1,1>& x,
                                                 const SystemSLAM& system,
                                                 std::size_t idxLandmark) const;

    Eigen::Vector3d predictDuckFeature(const Eigen::VectorXd& x,
                                       Eigen::MatrixXd& J,
                                       const SystemSLAM& system,
                                       std::size_t idxLandmark) const;

    // ---- Bundle prediction over a specific landmark index list (stacked [u v A ...]) ----
    template <typename Scalar>
    Eigen::Matrix<Scalar,-1,1> predictDuckBundle(const Eigen::Matrix<Scalar,-1,1>& x,
                                                 const SystemSLAM& system,
                                                 const std::vector<std::size_t>& idxLandmarks) const;

    Eigen::VectorXd predictDuckBundle(const Eigen::VectorXd& x,
                                      Eigen::MatrixXd& J,
                                      const SystemSLAM& system,
                                      const std::vector<std::size_t>& idxLandmarks) const;

private:
    Eigen::VectorXd A_;      // detected areas (length M; same column order as Y_)
    double rDuck_;           // duck radius in meters (Lab 3 prior)
    double sigmaA_;          // std-dev of area measurement (pixels^2 units)
};

#endif
