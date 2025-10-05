#include <cstddef>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemSLAM.h"
#include "rotation.hpp"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>


SystemSLAM::SystemSLAM(const GaussianInfo<double> & density)
    : SystemEstimator(density)
{}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemSLAM::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const
{
    assert(density.dim() == x.size());
    //
    //  dnu/dt =          0 + dwnu/dt
    // deta/dt = JK(eta)*nu +       0
    //   dm/dt =          0 +       0
    // \_____/   \________/   \_____/
    //  dx/dt  =    f(x)    +  dw/dt
    //
    //        [          0 ]
    // f(x) = [ JK(eta)*nu ]
    //        [          0 ] for all map states
    //
    //        [                    0 ]
    //        [                    0 ]
    // f(x) = [    Rnb(thetanb)*vBNb ]
    //        [ TK(thetanb)*omegaBNb ]
    //        [                    0 ] for all map states
    //
    Eigen::VectorXd f(x.size());
    f.setZero();
    // TODO: Implement in Assignment(s)
    // Extract states
    // Extract velocity states
    Eigen::Vector3d vBNb = x.segment<3>(0);      // Body translational velocity
    Eigen::Vector3d omegaBNb = x.segment<3>(3);  // Body angular velocity
    
    // Extract pose states (needed for kinematic transform)
    Eigen::Vector3d Thetanb = x.segment<3>(9);   // RPY Euler angles
    
    // Convert Euler angles to rotation matrix
    Eigen::Matrix3d Rnb = rpy2rot(Thetanb);
    
    // Kinematic transformation: dposition/dt = Rnb * vBNb
    f.segment<3>(6) = Rnb * vBNb;
    
    // Kinematic transformation: dTheta/dt = T * omegaBNb
    const double phi = Thetanb(0);
    const double theta = Thetanb(1);
    
    Eigen::Matrix3d T;
    T << 1.0, std::sin(phi)*std::tan(theta),  std::cos(phi)*std::tan(theta),
         0.0, std::cos(phi),                 -std::sin(phi),
         0.0, std::sin(phi)/std::cos(theta),  std::cos(phi)/std::cos(theta);
    
    f.segment<3>(9) = T * omegaBNb;

    // Landmark states: dm/dt = 0 (static landmarks)
    // Already set to zero above

    return f;
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemSLAM::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    //
    //  Jacobian J = df/dx
    //    
    //     [  0                  0 0 ]
    // J = [ JK d(JK(eta)*nu)/deta 0 ]
    //     [  0                  0 0 ]
    //
    // where JK(eta) = blkdiag(Rnb(Theta), T(Theta))
    //

    assert(density.dim() == x.size());
    
    using autodiff::dual;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;
    using autodiff::val;

    // OPTIMIZATION: f(x) only depends on [ν, η] (first 12 states), not landmarks
    // Therefore ∂f/∂mj = 0 for all landmarks (mathematically exact by Eq 4)
    
    const Eigen::Index nx = x.size();          // Total state dimension
    const Eigen::Index nVelPose = 12;          // [ν, η] dimension (6 + 6)
    
    // Extract ONLY the velocity and pose states for differentiation
    Eigen::VectorX<dual> x_vel_pose_dual = x.head(nVelPose).cast<dual>();
    
    // Define dynamics function for autodiff (operates only on first 12 states)
    auto f_dual = [&](const Eigen::VectorX<dual>& xd) -> Eigen::VectorX<dual> {
        // xd has dimension 12: [vBNb (3), omegaBNb (3), rBNn (3), Thetanb (3)]
        Eigen::VectorX<dual> fd(nx);  // Output has full dimension (including landmarks)
        fd.setZero();
        
        // Extract states
        Eigen::Vector3<dual> vBNb    = xd.segment<3>(0);  // Body translational velocity
        Eigen::Vector3<dual> omegaBNb = xd.segment<3>(3);  // Body angular velocity
        Eigen::Vector3<dual> Thetanb  = xd.segment<3>(9);  // RPY Euler angles
        
        // Rotation matrix (autodiff-compatible)
        Eigen::Matrix3<dual> Rnb = rpy2rot(Thetanb);
        
        // Position dynamics: ṙ = Rnb · vBNb
        fd.segment<3>(6) = Rnb * vBNb;
        
        // Orientation dynamics: Θ̇ = TK(Θ) · ωBNb
        const dual phi = Thetanb(0);
        const dual theta = Thetanb(1);
        
        Eigen::Matrix3<dual> T;
        T << dual(1.0), sin(phi)*tan(theta),  cos(phi)*tan(theta),
             dual(0.0), cos(phi),             -sin(phi),
             dual(0.0), sin(phi)/cos(theta),  cos(phi)/cos(theta);
        
        fd.segment<3>(9) = T * omegaBNb;
        
        // Landmark dynamics: ṁj = 0 (already zero from setZero())
        
        return fd;
    };
    
    // Compute Jacobian ONLY w.r.t. first 12 states
    Eigen::VectorX<dual> f_dual_out;
    Eigen::MatrixXd J_vel_pose = jacobian(f_dual, wrt(x_vel_pose_dual), at(x_vel_pose_dual), f_dual_out);
    // J_vel_pose is (nx × 12)
    
    // Construct full Jacobian by appending zero columns for landmarks
    J.resize(nx, nx);
    J.setZero();
    J.leftCols(nVelPose) = J_vel_pose;  // First 12 columns from autodiff
    // Remaining columns are zero (landmarks don't affect dynamics)
    
    // Extract function value as double
    Eigen::VectorXd f(nx);
    for (Eigen::Index i = 0; i < nx; ++i) {
        f(i) = val(f_dual_out(i));
    }
    
    return f;
}

// // Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
// Eigen::VectorXd SystemSLAM::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
// {
//     Eigen::VectorXd f = dynamics(t, x, u);

//     // Jacobian J = df/dx
//     //    
//     //     [  0                  0 0 ]
//     // J = [ JK d(JK(eta)*nu)/deta 0 ]
//     //     [  0                  0 0 ]
//     //
//     J.resize(f.size(), x.size());
//     J.setZero();
//     // TODO: Implement in Assignment(s)

//     // Extract states
//     Eigen::Vector3d vBNb = x.segment<3>(0);
//     Eigen::Vector3d omegaBNb = x.segment<3>(3);
//     Eigen::Vector3d Thetanb = x.segment<3>(9);
    
//     Eigen::Matrix3d Rnb = rpy2rot(Thetanb);
    
//     // J[6:8, 0:2] = Rnb (derivative of position w.r.t. velocity)
//     J.block<3,3>(6,0) = Rnb;
    
//     // J[6:8, 9:11] = d(Rnb*vBNb)/dTheta (numerical differentiation for robustness)
//     const double eps = 1e-7;
//     for (int i = 0; i < 3; ++i) {
//         Eigen::Vector3d Theta_plus = Thetanb;
//         Theta_plus(i) += eps;
//         Eigen::Matrix3d Rnb_plus = rpy2rot(Theta_plus);
//         J.block<3,1>(6, 9+i) = (Rnb_plus * vBNb - Rnb * vBNb) / eps;
//     }
    
//     // J[9:11, 3:5] = T (derivative of orientation w.r.t. angular velocity)
//     const double phi = Thetanb(0);
//     const double theta = Thetanb(1);
//     Eigen::Matrix3d T;
//     T << 1.0, std::sin(phi)*std::tan(theta),  std::cos(phi)*std::tan(theta),
//          0.0, std::cos(phi),                 -std::sin(phi),
//          0.0, std::sin(phi)/std::cos(theta),  std::cos(phi)/std::cos(theta);
//     J.block<3,3>(9,3) = T;
    
//     // J[9:11, 9:11] = d(T*omega)/dTheta (numerical differentiation)
//     for (int i = 0; i < 3; ++i) {
//         Eigen::Vector3d Theta_plus = Thetanb;
//         Theta_plus(i) += eps;
//         const double phi_p = Theta_plus(0);
//         const double theta_p = Theta_plus(1);
//         Eigen::Matrix3d T_plus;
//         T_plus << 1.0, std::sin(phi_p)*std::tan(theta_p),  std::cos(phi_p)*std::tan(theta_p),
//                   0.0, std::cos(phi_p),                    -std::sin(phi_p),
//                   0.0, std::sin(phi_p)/std::cos(theta_p),  std::cos(phi_p)/std::cos(theta_p);
//         J.block<3,1>(9, 9+i) = (T_plus * omegaBNb - T * omegaBNb) / eps;
//     }
//     return f;
// }

Eigen::VectorXd SystemSLAM::input(double t, const Eigen::VectorXd & x) const
{
    return Eigen::VectorXd(0);
}

GaussianInfo<double> SystemSLAM::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
    Eigen::MatrixXd SQ(6, 6);
    SQ.setZero();

    // Process noise parameters (TUNABLE - start here and adjust based on drift)
    // These control how much the filter trusts the process model vs measurements
    const double qv = 0.0001;  // Translational velocity noise (m/s^1.5)
    const double qw = 0.0002;  // Angular velocity noise (rad/s^1.5)
    
    // TODO: Assignment(s)
    // Diagonal noise (independent noise on each velocity component)
    SQ(0,0) = std::sqrt(qv);  // x-velocity noise
    SQ(1,1) = std::sqrt(qv);  // y-velocity noise
    SQ(2,2) = std::sqrt(qv);  // z-velocity noise
    SQ(3,3) = std::sqrt(qw);  // roll rate noise
    SQ(4,4) = std::sqrt(qw);  // pitch rate noise
    SQ(5,5) = std::sqrt(qw);  // yaw rate noise

    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return GaussianInfo<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemSLAM::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    // Noise only affects velocity states (indices 0-5)
    std::vector<Eigen::Index> idxQ{0, 1, 2, 3, 4, 5};
    // TODO: Assignment(s)
    return idxQ;
}

cv::Mat & SystemSLAM::view()
{
    return view_;
};

const cv::Mat & SystemSLAM::view() const
{
    return view_;
};

GaussianInfo<double> SystemSLAM::bodyPositionDensity() const
{
    return density.marginal(Eigen::seqN(6, 3));
}

GaussianInfo<double> SystemSLAM::bodyOrientationDensity() const
{
    return density.marginal(Eigen::seqN(9, 3));
}

GaussianInfo<double> SystemSLAM::bodyTranslationalVelocityDensity() const
{
    return density.marginal(Eigen::seqN(0, 3));
}

GaussianInfo<double> SystemSLAM::bodyAngularVelocityDensity() const
{
    return density.marginal(Eigen::seqN(3, 3));
}

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

Eigen::Vector3d SystemSLAM::cameraPosition(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> rCNn_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraPosition<autodiff::dual>, wrt(x_dual), at(camera, x_dual), rCNn_dual);
    return rCNn_dual.cast<double>();
};

GaussianInfo<double> SystemSLAM::cameraPositionDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraPosition(camera, x, J); };
    return density.affineTransform(f);
}

Eigen::Vector3d SystemSLAM::cameraOrientationEuler(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> Thetanc_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraOrientationEuler<autodiff::dual>, wrt(x_dual), at(camera, x_dual), Thetanc_dual);
    return Thetanc_dual.cast<double>();
};

GaussianInfo<double> SystemSLAM::cameraOrientationEulerDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraOrientationEuler(camera, x, J); };
    return density.affineTransform(f);    
}

GaussianInfo<double> SystemSLAM::landmarkPositionDensity(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    return density.marginal(Eigen::seqN(idx, 3));
}
