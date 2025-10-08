#ifndef MEASUREMENTSLAMDUCKBUNDLE_H
#define MEASUREMENTSLAMDUCKBUNDLE_H

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include <numeric>
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
    Eigen::VectorXd simulate(const Eigen::VectorXd& x, const SystemEstimator& system) const override
    {
        // This function is required to be defined, but not critical for the main SLAM loop.
        // It's useful for testing and generating synthetic data.
        const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
        std::vector<std::size_t> idxLandmarks(sys.numberLandmarks());
        std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);
        Eigen::MatrixXd J_dummy; // Jacobian is not used for simulation
        return predictDuckBundle(x, J_dummy, sys, idxLandmarks);
    }

    // Likelihoods over stacked [u v A ...] for associated pairs
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system) const override;
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g) const override;
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g, Eigen::MatrixXd& H) const override;

    // Plot needs this to colour landmarks blue when matched this frame
    bool isEffectivelyAssociated(std::size_t j) const {
        return j < is_effectively_associated_.size() && is_effectively_associated_[j];
    }

    // Densities required by base API (used by Plot for 2D ellipses and SNN for association)
    GaussianInfo<double> predictFeatureDensity(const SystemSLAM& system, std::size_t idxLandmark) const override;
    GaussianInfo<double> predictFeatureBundleDensity(const SystemSLAM& system, const std::vector<std::size_t>& idxLandmarks) const override;
    
    // Expose centroids for SNN association (shape 2xN)
    const Eigen::Matrix<double,2,Eigen::Dynamic>& Y() const { return Yuv_; }

    // Association uses 2D (u,v) density; pass that to SNN
    const std::vector<int>& associate(const SystemSLAM& system, const std::vector<std::size_t>& idxLandmarks) override;

    // Association results (landmark index -> feature index or -1)
    const std::vector<int>& idxFeatures() const { return idxFeatures_; }

protected:
    void update(SystemBase& system) override;

    // Templated prediction function for autodiff
    template<typename Scalar>
    Eigen::Vector3<Scalar> predictDuckT(const Eigen::VectorX<Scalar>& x,
                                        const SystemSLAM& system,
                                        std::size_t idxLandmark) const;
    
    // Overload that provides the Jacobian via autodiff
    Eigen::Vector3d predictDuck(const Eigen::VectorXd& x, Eigen::MatrixXd& J, const SystemSLAM& system, std::size_t idxLandmark) const;

    // Bundle prediction (3x per landmark): stacks [u v A] for all landmarks
    Eigen::VectorXd predictDuckBundle(const Eigen::VectorXd& x, Eigen::MatrixXd& J, const SystemSLAM& system,
                                      const std::vector<std::size_t>& idxLandmarks) const;

private:
    // ---- Helpers (private) ----
    GaussianInfo<double> predictCentroidBundleDensity(const SystemSLAM& system,
                                                      const std::vector<std::size_t>& idxLandmarks) const;

    // ---- Measurement storage ----
    Eigen::Matrix<double,2,Eigen::Dynamic> Yuv_;  // centroids (2xN) for this frame
    Eigen::VectorXd                        A_;      // areas (N)

    // ---- Noise/tuning ----
    double sigma_c_;    // pixel std for (u,v)
    double sigma_a_;    // area std (pixels^2)
    double duck_r_m_;   // physical duck radius [m]
    double fx_, fy_;    // cached intrinsics for area model

    // ---- Association state ----
    std::vector<int>  idxFeatures_;               // LM j -> feature index (-1 if none)
    std::vector<bool> is_effectively_associated_;   // for Plot colouring
};


// Templated prediction function (the core logic)
// This must be in the header file to be available for instantiation by other files and autodiff.
template<typename Scalar>
Eigen::Vector3<Scalar>
MeasurementSLAMDuckBundle::predictDuckT(const Eigen::VectorX<Scalar>& x,
                                        const SystemSLAM& system,
                                        std::size_t idxLandmark) const
{
    // Get camera pose from the state vector x
    Pose<Scalar> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x);
    Tnc.rotationMatrix    = system.cameraOrientation(camera_, x);

    // Get the landmark's 3D position from the state vector x
    const std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rLNn = x.template segment<3>(idx);

    // Transform the landmark's position from world coordinates to camera coordinates
    const Eigen::Matrix3<Scalar> Rcn = Tnc.rotationMatrix.transpose();
    const Eigen::Vector3<Scalar> rLCc = Rcn * (rLNn - Tnc.translationVector);

    // Project the 3D point in camera coordinates to 2D pixel coordinates
    const Eigen::Vector2<Scalar> uv = camera_.vectorToPixel(rLCc);

    // Calculate the predicted area based on the landmark's depth
    const Scalar depth = rLCc.norm();
    const Scalar A = (Scalar(fx_) * Scalar(fy_) * Scalar(std::numbers::pi) * Scalar(duck_r_m_) * Scalar(duck_r_m_)) / (depth * depth);

    // Assemble the 3D measurement vector [u, v, Area]
    Eigen::Vector3<Scalar> h;
    h << uv(0), uv(1), A;
    return h;
}

#endif // MEASUREMENTSLAMDUCKBUNDLE_H

