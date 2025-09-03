#include <cmath>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Gaussian.hpp"
#include "MeasurementGaussianLikelihood.h"
#include "MeasurementRADAR.h"

const double MeasurementRADAR::r1 = 5000;    // Horizontal position of sensor [m]
const double MeasurementRADAR::r2 = 5000;    // Vertical position of sensor [m]

MeasurementRADAR::MeasurementRADAR(double time, const Eigen::VectorXd & y)
    : MeasurementGaussianLikelihood(time, y)
{
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRTINV;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;

    // updateMethod_ = UpdateMethod::AFFINE;
    // updateMethod_ = UpdateMethod::GAUSSNEWTON;
    // updateMethod_ = UpdateMethod::LEVELBERGMARQUARDT;
}

MeasurementRADAR::MeasurementRADAR(double time, const Eigen::VectorXd & y, int verbosity)
    : MeasurementGaussianLikelihood(time, y, verbosity)
{
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRTINV;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;

    // updateMethod_ = UpdateMethod::AFFINE;
    // updateMethod_ = UpdateMethod::GAUSSNEWTON;
    // updateMethod_ = UpdateMethod::LEVELBERGMARQUARDT;
}

MeasurementRADAR::~MeasurementRADAR() = default;

std::string MeasurementRADAR::getProcessString() const
{
    return "RADAR measurement update:";
}

// Evaluate h(x) from the measurement model y = h(x) + v
Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd h(1);
    // TODO
    return h;
}

// Evaluate h(x) and its Jacobian J = dh/fx from the measurement model y = h(x) + v
Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::MatrixXd & dhdx) const
{
    Eigen::VectorXd h = predict(x, system);

    //              dh_i
    // dhdx(i, j) = ----
    //              dx_j
    dhdx.resize(h.size(), x.size());
    dhdx.setZero();
    // TODO: Set non-zero elements of dhdx

    return h;
}

Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::MatrixXd & dhdx, Eigen::Tensor<double, 3> & d2hdx2) const
{
    Eigen::VectorXd h = predict(x, system, dhdx);

    //                    d^2 h_i     d 
    // d2hdx2(i, j, k) = --------- = ---- dhdx(i, j)
    //                   dx_j dx_k   dx_k
    d2hdx2.resize(h.size(), x.size(), x.size());
    d2hdx2.setZero();
    // TODO: Set non-zero elements of d2hdx2

    return h;
}

Gaussian<double> MeasurementRADAR::noiseDensity(const SystemEstimator & system) const
{
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    Eigen::MatrixXd SR(1, 1);
    // TODO
    return Gaussian<double>::fromSqrtMoment(SR);
}

