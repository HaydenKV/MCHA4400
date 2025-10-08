#ifndef MEASUREMENTSLAMDUCKBUNDLE_H
#define MEASUREMENTSLAMDUCKBUNDLE_H

#include <Eigen/Core>
#include <vector>
#include "MeasurementSLAM.h"
#include "Camera.h"
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"

class MeasurementSLAMDuckBundle : public MeasurementSLAM
{
public:
    MeasurementSLAMDuckBundle(double time,
                              const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                              const Eigen::VectorXd& A,
                              const Camera& camera,
                              double sigma_c_px,
                              double sigma_a_px2,
                              double duck_radius_m);

    MeasurementSLAM* clone() const override;

    // Base Measurement interface
    Eigen::VectorXd simulate(const Eigen::VectorXd& x, const SystemEstimator& system) const override;

    // Likelihoods over stacked [u v A ...] for associated pairs
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system) const override;
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g) const override;
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g, Eigen::MatrixXd& H) const override;

    bool isEffectivelyAssociated(std::size_t j) const {
        return j < is_effectively_associated_.size() && is_effectively_associated_[j];
    }

    // Densities required by base API (used by Plot for 2D ellipses)
    GaussianInfo<double> predictFeatureDensity(const SystemSLAM& system, std::size_t idxLandmark) const override;
    GaussianInfo<double> predictFeatureBundleDensity(const SystemSLAM& system, const std::vector<std::size_t>& idxLandmarks) const override;
    
    const Eigen::Matrix<double,2,Eigen::Dynamic>& Y() const { return Yuv_; }

    const std::vector<int>& associate(const SystemSLAM& system, const std::vector<std::size_t>& idxLandmarks) override;

    const std::vector<int>& idxFeatures() const { return idxFeatures_; }

protected:
    void update(SystemBase& system) override;

    Eigen::Vector3d predictDuck(const Eigen::VectorXd& x, Eigen::MatrixXd& J, const SystemSLAM& system, std::size_t idxLandmark) const;

    // This function is now private and doesn't use templates
    Eigen::VectorXd predictDuckBundle(const Eigen::VectorXd& x, Eigen::MatrixXd& J, const SystemSLAM& system,
                                      const std::vector<std::size_t>& idxLandmarks) const;

private:
    // ---- Helpers (private) ----
    Eigen::Vector3d predictDuckT(const Eigen::VectorXd& x, const SystemSLAM& system, std::size_t idxLandmark) const;

    GaussianInfo<double> predictCentroidBundleDensity(const SystemSLAM& system,
                                                      const std::vector<std::size_t>& idxLandmarks) const;

    // ---- Measurement storage ----
    Eigen::Matrix<double,2,Eigen::Dynamic> Yuv_;
    Eigen::VectorXd                        A_;

    // ---- Noise/tuning ----
    double sigma_c_;
    double sigma_a_;
    double duck_r_m_;
    double fx_, fy_;

    // ---- Association state ----
    std::vector<int>  idxFeatures_;
    std::vector<bool> is_effectively_associated_;
};

#endif // MEASUREMENTSLAMDUCKBUNDLE_H