#include <doctest/doctest.h>
#include <Eigen/Core>
#include "../../src/GaussianInfo.hpp"
#include "../../src/SystemSLAM.h"
#include "../../src/SystemSLAMPoseLandmarks.h"
#include "../../src/SystemSLAMPointLandmarks.h"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision,0,", ",";\n","","","[","]")));
#endif

// Helper: n = 12 + 6·L (pose-landmarks). Order: [v_B, ω_B, r_BN^n, Θ, {landmarks...}].
static GaussianInfo<double> makeDensity(int L)
{
    const int n = 12 + 6*L;
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
    mu.segment<3>(6) << 1,2,3; // r_BN^n (non-zero)
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n);
    return GaussianInfo<double>::fromSqrtMoment(mu, S);
}

static Eigen::VectorXd makeStateFromMu(const GaussianInfo<double>& d) { return d.mean(); }

// -------------------------------------------------------------------------------------------------
// Zero-motion invariants  (assignment: process model)
//   ṙ_BN^n = R_nb(Θ) v_B ,   Θ̇ = T(Θ) ω_B ,   static map
// Test: with v_B=0, ω_B=0, Θ=0 ⇒ v̇=0, ω̇=0, ṙ=0, Θ̇=0 and landmark̇=0.
// -------------------------------------------------------------------------------------------------
SCENARIO("SystemSLAM dynamics invariants at zero body motion")
{
    GIVEN("Theta=0, v=0, w=0 and arbitrary position/landmarks")
    {
        auto density = makeDensity(/*L=*/2);
        SystemSLAMPoseLandmarks sys(density);
        Eigen::VectorXd x = makeStateFromMu(density);

        x.segment<3>(0).setZero(); // v_B
        x.segment<3>(3).setZero(); // ω_B
        x.segment<3>(9).setZero(); // Θ

        WHEN("f(x) is evaluated")
        {
            Eigen::VectorXd f = sys.dynamics(0.0, x, Eigen::VectorXd());

            THEN("Base derivatives are zero")
            {
                REQUIRE(f.size() == x.size());
                CHECK(f.segment<3>(0).isZero(0));  // v̇_B
                CHECK(f.segment<3>(3).isZero(0));  // ω̇_B
                CHECK(f.segment<3>(6).isZero(0));  // ṙ_BN^n
                CHECK(f.segment<3>(9).isZero(0));  // Θ̇
            }

            THEN("Landmark derivatives are zero")
            {
                CHECK(f.segment<6>(12 + 0*6).isZero(0));
                CHECK(f.segment<6>(12 + 1*6).isZero(0));
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Θ=0 ⇒  R_nb(0)=I ,  T(0)=I   (assignment: attitude and Euler-rate maps)
// Test: ṙ = v_B  and  Θ̇ = ω_B at Θ=0.
// -------------------------------------------------------------------------------------------------
SCENARIO("SystemSLAM kinematics at zero attitude (R=I, T=I)")
{
    GIVEN("Theta=0 with nonzero body velocities")
    {
        auto density = makeDensity(/*L=*/1);
        SystemSLAMPoseLandmarks sys(density);
        Eigen::VectorXd x = makeStateFromMu(density);

        x.segment<3>(0) << 1.0, 0.5, -0.2;   // v_B
        x.segment<3>(3) << 0.1, -0.2, 0.3;  // ω_B
        x.segment<3>(9).setZero();          // Θ

        WHEN("f(x) is evaluated")
        {
            Eigen::VectorXd f = sys.dynamics(0.0, x, Eigen::VectorXd());

            THEN("ṙ equals v_B")
            {
                CHECK(f.segment<3>(6).isApprox(x.segment<3>(0)));
            }

            THEN("Θ̇ equals ω_B")
            {
                CHECK(f.segment<3>(9).isApprox(x.segment<3>(3)));
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Static map (assignment: landmarks do not evolve in the process model)
// Test: landmark̇ = 0 for all landmarks for arbitrary (v_B, ω_B).
// -------------------------------------------------------------------------------------------------
SCENARIO("SystemSLAM: landmark stationarity regardless of platform motion")
{
    GIVEN("Nonzero v and w; arbitrary landmarks")
    {
        auto density = makeDensity(/*L=*/3);
        SystemSLAMPoseLandmarks sys(density);
        Eigen::VectorXd x = makeStateFromMu(density);

        x.segment<3>(0) << 0.2, -1.0, 0.7;
        x.segment<3>(3) << -0.4, 0.9, -0.1;

        WHEN("f(x) is evaluated")
        {
            Eigen::VectorXd f = sys.dynamics(0.0, x, Eigen::VectorXd());

            THEN("Each landmark derivative is zero")
            {
                CHECK(f.segment<6>(12 + 0*6).isZero(0));
                CHECK(f.segment<6>(12 + 1*6).isZero(0));
                CHECK(f.segment<6>(12 + 2*6).isZero(0));
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Point-landmark indexing (assignment: 3D per point, order = 12 + 3k)
// Test: numberLandmarks() and landmarkPositionIndex(k) = 12 + 3k.
// -------------------------------------------------------------------------------------------------
SCENARIO("SystemSLAMPointLandmarks indexing and counts")
{
    GIVEN("A Gaussian density with body + 2 point landmarks (3D each)")
    {
        const int nx = 12 + 2*3;
        auto g = GaussianInfo<double>::fromSqrtMoment(
            Eigen::VectorXd::Zero(nx), Eigen::MatrixXd::Identity(nx,nx)
        );
        SystemSLAMPointLandmarks sys(g);

        THEN("Indices and counts are correct")
        {
            CHECK(sys.numberLandmarks() == 2);
            CHECK(sys.landmarkPositionIndex(0) == 12);
            CHECK(sys.landmarkPositionIndex(1) == 15);
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Pose-landmark append (assignment: +6 per pose landmark, appended after base 12)
// Test: appendLandmark → new landmark has index 0; position index = 12.
// -------------------------------------------------------------------------------------------------
SCENARIO("SystemSLAMPoseLandmarks indexing and appendLandmark")
{
    GIVEN("An empty pose-landmark SLAM system")
    {
        const int nx = 12;
        auto g = GaussianInfo<double>::fromSqrtMoment(
            Eigen::VectorXd::Zero(nx), Eigen::MatrixXd::Identity(nx,nx)
        );
        SystemSLAMPoseLandmarks sys(g);

        THEN("appendLandmark increments size and indices as expected")
        {
            Eigen::Vector3d r(1,2,3);
            Eigen::Vector3d th(0.1,0.2,0.3);
            Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Identity();

            std::size_t idx = sys.appendLandmark(r, th, Sj);

            CHECK(idx == 0);
            CHECK(sys.numberLandmarks() == 1);
            CHECK(sys.landmarkPositionIndex(0) == 12);
        }
    }
}
