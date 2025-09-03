#include <cmath>
#include <cassert>
#include <Eigen/Core>
#include "Gaussian.hpp"
#include "SystemEstimator.h"
#include "SystemBallistic.h"

const double SystemBallistic::p0 = 101.325e3;            // Air pressure at sea level [Pa]
const double SystemBallistic::M  = 0.0289644;            // Molar mass of dry air [kg/mol]
const double SystemBallistic::R  = 8.31447;              // Gas constant [J/(mol.K)]
const double SystemBallistic::L  = 0.0065;               // Temperature gradient [K/m]
const double SystemBallistic::T0 = 288.15;               // Temperature at sea level [K]
const double SystemBallistic::g  = 9.81;                 // Acceleration due to gravity [m/s^2]

SystemBallistic::SystemBallistic(const Gaussian<double> & density)
    : SystemEstimator(density)
{}

Gaussian<double> SystemBallistic::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
    Eigen::MatrixXd SQ(2, 2);
    // TODO
    
    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return Gaussian<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemBallistic::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    std::vector<Eigen::Index> idxQ;
    // TODO: Continuous-time process noise in 2nd and 3rd state equations
    return idxQ;
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemBallistic::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const
{
    Eigen::VectorXd f(x.size());
    // TODO: Set f
    return f;
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemBallistic::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(t, x, u);

    J.resize(f.size(), x.size());
    // TODO: Set J

    return f;
}

Eigen::VectorXd SystemBallistic::input(double t, const Eigen::VectorXd & x) const
{
    return Eigen::VectorXd(0);
}

