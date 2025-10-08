#include "MeasurementSLAMDuckBundle.h"
#include <cassert>
#include <numeric>
#include <numbers>
#include "SystemEstimator.h"
#include "SystemBase.h"
#include "Pose.hpp"
#include "association_util.h"

// Autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace {
    // Helper function to build the diagonal weighting matrix for the likelihood.
    // Scoped to this file only to prevent redefinition errors.
    static inline void buildInvWeights(std::size_t k, double sc, double sa, Eigen::VectorXd& w)
    {
        w.resize(3 * k);
        const double inv_var_c = 1.0 / (sc * sc);
        const double inv_var_a = 1.0 / (sa * sa);
        for (std::size_t i = 0; i < k; ++i) {
            const std::size_t r = 3 * i;
            w(r+0) = inv_var_c;
            w(r+1) = inv_var_c;
            w(r+2) = inv_var_a;
        }
    }
}

// Constructor
MeasurementSLAMDuckBundle::MeasurementSLAMDuckBundle(double time,
                                                     const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                                                     const Eigen::VectorXd& A,
                                                     const Camera& camera,
                                                     double sigma_c_px,
                                                     double sigma_a_px2,
                                                     double duck_radius_m)
: MeasurementSLAM(time, camera)
, Yuv_(Yuv)
, A_(A)
, sigma_c_(sigma_c_px)
, sigma_a_(sigma_a_px2)
, duck_r_m_(duck_radius_m)
, fx_(camera.cameraMatrix.at<double>(0,0))
, fy_(camera.cameraMatrix.at<double>(1,1))
{
    assert(Yuv_.cols() == A_.size() && "DuckBundle: Y(2xN) and A(N) must match");
    assert(sigma_c_ > 0.0 && sigma_a_ > 0.0 && duck_r_m_ > 0.0);
    // Use a full Newton method for accuracy, especially during initialization
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM* MeasurementSLAMDuckBundle::clone() const
{
    auto* m = new MeasurementSLAMDuckBundle(time_, Yuv_, A_, camera_,
                                            sigma_c_, sigma_a_, duck_r_m_);
    m->idxFeatures_ = idxFeatures_;
    m->is_effectively_associated_ = is_effectively_associated_;
    return m;
}

// Predicts a single landmark and computes Jacobian via AUTODIFF
Eigen::Vector3d
MeasurementSLAMDuckBundle::predictDuck(const Eigen::VectorXd& x,
                                       Eigen::MatrixXd& J,
                                       const SystemSLAM& system,
                                       std::size_t idxLandmark) const
{
    using autodiff::dual;
    using autodiff::wrt;
    using autodiff::at;
    using autodiff::val;
    using autodiff::jacobian;

    auto h_func = [&](const Eigen::VectorX<dual>& x_dual) -> Eigen::Vector3<dual> {
        return predictDuckT<dual>(x_dual, system, idxLandmark);
    };

    Eigen::VectorX<dual> x_dual = x.cast<dual>();
    Eigen::Vector3<dual> h_dual;
    J = jacobian(h_func, wrt(x_dual), at(x_dual), h_dual);

    return h_dual.cast<double>();
}

// Stacks predictions for a bundle of landmarks
Eigen::VectorXd
MeasurementSLAMDuckBundle::predictDuckBundle(const Eigen::VectorXd& x,
                                             Eigen::MatrixXd& J,
                                             const SystemSLAM& system,
                                             const std::vector<std::size_t>& idxLandmarks) const
{
    const std::size_t nL = idxLandmarks.size();
    const std::size_t nx = system.density.dim();

    Eigen::VectorXd h(3 * nL);
    J.resize(3 * nL, nx);

    for (std::size_t i = 0; i < nL; ++i) {
        Eigen::MatrixXd Ji;
        Eigen::Vector3d hi = predictDuck(x, Ji, system, idxLandmarks[i]);
        h.segment<3>(3 * i) = hi;
        J.middleRows(3 * i, 3) = Ji;
    }
    return h;
}

// --- Density and Association Functions ---

GaussianInfo<double> MeasurementSLAMDuckBundle::predictCentroidBundleDensity(
    const SystemSLAM& system,
    const std::vector<std::size_t>& idxLandmarks) const
{
    const std::size_t nx = system.density.dim();
    const std::size_t L  = idxLandmarks.size();
    const std::size_t ny = 2 * L;

    const auto func = [&](const Eigen::VectorXd& xv, Eigen::MatrixXd& Ja)
    {
        const Eigen::VectorXd x = xv.head(nx);
        const Eigen::VectorXd v = xv.tail(ny);
        Eigen::VectorXd h2D(ny);
        Eigen::MatrixXd J2D(ny, nx);

        for (std::size_t i = 0; i < L; ++i) {
            Eigen::MatrixXd Ji;
            Eigen::Vector3d hi = predictDuck(x, Ji, system, idxLandmarks[i]);
            h2D.segment<2>(2*i) = hi.head<2>();
            J2D.block(2*i, 0, 2, nx) = Ji.topRows(2);
        }
        Ja.resize(ny, nx + ny);
        Ja << J2D, Eigen::MatrixXd::Identity(ny, ny);
        return h2D + v;
    };

    Eigen::MatrixXd S_centroid_noise = Eigen::MatrixXd::Identity(ny, ny) * sigma_c_;
    auto pv  = GaussianInfo<double>::fromSqrtMoment(S_centroid_noise);
    auto pxv = system.density * pv;
    return pxv.affineTransform(func);
}

GaussianInfo<double>
MeasurementSLAMDuckBundle::predictFeatureDensity(const SystemSLAM& system, std::size_t idxLandmark) const
{
    return predictCentroidBundleDensity(system, {idxLandmark});
}

GaussianInfo<double>
MeasurementSLAMDuckBundle::predictFeatureBundleDensity(const SystemSLAM& system, const std::vector<std::size_t>& idxLandmarks) const
{
    return predictCentroidBundleDensity(system, idxLandmarks);
}

const std::vector<int>& MeasurementSLAMDuckBundle::associate(
    const SystemSLAM& system,
    const std::vector<std::size_t>& idxLandmarks)
{
    const std::size_t nL = system.numberLandmarks();
    is_effectively_associated_.assign(nL, false);
    idxFeatures_.assign(nL, -1);

    if (idxLandmarks.empty() || Y().cols() == 0) return idxFeatures_;

    GaussianInfo<double> centroidBundleDensity = predictCentroidBundleDensity(system, idxLandmarks);
    snn(system, centroidBundleDensity, idxLandmarks, Y(), camera_, idxFeatures_);

    for (std::size_t j = 0; j < idxLandmarks.size(); ++j) {
        // Here, j is an index into the idxLandmarks vector
        // The value idxLandmarks[j] is the actual landmark index in the state vector
        // The value idxFeatures_[j] is the feature index associated with landmark idxLandmarks[j]
        if (idxFeatures_[j] >= 0) {
             is_effectively_associated_[idxLandmarks[j]] = true;
        }
    }
    return idxFeatures_;
}

void MeasurementSLAMDuckBundle::update(SystemBase& systemBase)
{
    SystemSLAM& sys = dynamic_cast<SystemSLAM&>(systemBase);
    // Landmark initialization is now handled in visualNavigation.cpp
    // This function will just perform the update on the existing map.
    std::vector<std::size_t> all_landmarks(sys.numberLandmarks());
    std::iota(all_landmarks.begin(), all_landmarks.end(), 0);
    // Run data association
    associate(sys, all_landmarks);
    // Perform the measurement update using the base class method
    Measurement::update(systemBase);
}

// --- Likelihood Functions (Corrected Implementation) ---

double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    
    std::vector<std::size_t> associated_landmarks;
    std::vector<int> associated_features;
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j) {
        if (idxFeatures_[j] >= 0) {
            associated_landmarks.push_back(j);
            associated_features.push_back(idxFeatures_[j]);
        }
    }
    
    const std::size_t k = associated_landmarks.size();
    if (k == 0) return 0.0;

    Eigen::VectorXd y_stacked(3 * k);
    for (std::size_t i = 0; i < k; ++i) {
        y_stacked.segment<2>(3*i) = Yuv_.col(associated_features[i]);
        y_stacked(3*i + 2)        = A_(associated_features[i]);
    }

    Eigen::VectorXd h(3 * k);
    for (size_t i = 0; i < k; ++i) {
        h.segment<3>(3 * i) = predictDuckT<double>(x, sys, associated_landmarks[i]);
    }
    
    Eigen::VectorXd weights;
    buildInvWeights(k, sigma_c_, sigma_a_, weights);
    const Eigen::VectorXd r = y_stacked - h;
    
    double ll = 0.0;
    for (int i = 0; i < r.size(); ++i) {
        ll += -0.5 * weights(i) * r(i) * r(i);
    }
    return ll;
}


double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system,
                                         Eigen::VectorXd& g) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    
    std::vector<std::size_t> associated_landmarks;
    std::vector<int> associated_features;
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j) {
        if (idxFeatures_[j] >= 0) {
            associated_landmarks.push_back(j);
            associated_features.push_back(idxFeatures_[j]);
        }
    }
    
    const std::size_t k = associated_landmarks.size();
    g.setZero(x.size());
    
    if (k > 0) {
        Eigen::VectorXd y_stacked(3 * k);
        for (std::size_t i = 0; i < k; ++i) {
            y_stacked.segment<2>(3*i) = Yuv_.col(associated_features[i]);
            y_stacked(3*i + 2)        = A_(associated_features[i]);
        }
        
        Eigen::MatrixXd J;
        Eigen::VectorXd h = predictDuckBundle(x, J, sys, associated_landmarks);
        
        Eigen::VectorXd weights;
        buildInvWeights(k, sigma_c_, sigma_a_, weights);
        const Eigen::VectorXd r = y_stacked - h;

        g.noalias() = J.transpose() * (weights.asDiagonal() * r);
    }
    
    return logLikelihood(x, system);
}

double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system,
                                         Eigen::VectorXd& g,
                                         Eigen::MatrixXd& H) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    
    std::vector<std::size_t> associated_landmarks;
    std::vector<int> associated_features;
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j) {
        if (idxFeatures_[j] >= 0) {
            associated_landmarks.push_back(j);
            associated_features.push_back(idxFeatures_[j]);
        }
    }
    
    const std::size_t k = associated_landmarks.size();
    g.setZero(x.size());
    H = Eigen::MatrixXd::Zero(x.size(), x.size());

    if (k > 0) {
        Eigen::MatrixXd J;
        // This call computes the Jacobian J needed for g and H
        predictDuckBundle(x, J, sys, associated_landmarks);
        
        Eigen::VectorXd weights;
        buildInvWeights(k, sigma_c_, sigma_a_, weights);

        // H_gn = -J^T * W * J (Gauss-Newton approximation of the Hessian)
        H.noalias() = -J.transpose() * weights.asDiagonal() * J;
    }
    
    // Now that H is computed, we can call the (x, g) overload which will correctly compute g
    // and the final scalar log-likelihood. This preserves the tested pattern.
    return logLikelihood(x, system, g);
}

// Explicitly instantiate the template function for the autodiff scalar type.
// This is necessary because the definition is in the header.
template Eigen::Vector3<autodiff::dual>
MeasurementSLAMDuckBundle::predictDuckT<autodiff::dual>(const Eigen::VectorX<autodiff::dual>&,
                                                        const SystemSLAM&,
                                                        std::size_t) const;

