#ifndef MEASUREMENTSLAMUNIQUETAGBUNDLE_H
#define MEASUREMENTSLAMUNIQUETAGBUNDLE_H

#include <vector>
#include <array>
#include <Eigen/Core>
#include "MeasurementSLAM.h"
#include "SystemSLAMPoseLandmarks.h"
#include "rotation.hpp"

struct TagDetection
{
    int id = -1;
    // 4 corners in image pixels, ordered TL, TR, BR, BL (OpenCV ArUco order)
    // corners[i] = {u, v}
    std::array<Eigen::Vector2d, 4> corners{};
};

class MeasurementSLAMUniqueTagBundle : public MeasurementSLAM
{
public:
    // Y is 2x(4*Ntags): [TL,TR,BR,BL | TL,TR,BR,BL | ...] column-stacked
    MeasurementSLAMUniqueTagBundle(double time,
                                   const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
                                   const Camera& camera,
                                   const std::vector<TagDetection>& detections,
                                   double tagSizeMeters = 0.16);

    MeasurementSLAM * clone() const override { return new MeasurementSLAMUniqueTagBundle(*this); }

    // Predict a single "feature" for Plot purposes: we use the tag CENTER (2D)
    GaussianInfo<double> predictFeatureDensity(const SystemSLAM& system, std::size_t idxLandmark) const override;

    // Predict stacked centers for an arbitrary set (mostly unused by Plot, provided for completeness)
    GaussianInfo<double> predictFeatureBundleDensity(const SystemSLAM& system,
                                                     const std::vector<std::size_t>& idxLandmarks) const override;

    // Association is trivial by ID at the tag level; here we return a no-op vector
    // (We’ll do true ID→landmark management inside update(...) in Stage-2)
    const std::vector<int>& associate(const SystemSLAM& system,
                                      const std::vector<std::size_t>& idxLandmarks) override;

    // --- Stage-1 stubs so the class is concrete ---
    Eigen::VectorXd simulate(const Eigen::VectorXd& x,
                             const SystemEstimator& system) const override;
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
    // Stage-1: Map management disabled; just call base update when you enable fusion
    void update(SystemBase& system) override;

private:
    // Returns tag center pixel projection for the given POSE-landmark index
    template <typename Scalar>
    Eigen::Vector2<Scalar> predictTagCenter(const Eigen::VectorX<Scalar>& x,
                                            const SystemSLAM& system,
                                            std::size_t idxPoseLandmark) const;

    // Returns the 4 world corners of a tag given its pose (center rLNn + orientation Rln)
    template <typename Scalar>
    std::array<Eigen::Vector3<Scalar>, 4> tagCornersWorld(const Eigen::Vector3<Scalar>& rLNn,
                                                          const Eigen::Matrix3<Scalar>& Rln) const;

private:
    Eigen::Matrix<double,2,Eigen::Dynamic> Y_;   // 2 x (4*Ntags) pixel columns
    std::vector<TagDetection> detections_;       // one per tag this frame
    double sigma_ = 1.0;                         // pixel stdev for visualisation
    double tagSize_ = 0.16;                      // meters, edge length
    mutable std::vector<int> idxFeatures_;       // required by interface
};

#endif
