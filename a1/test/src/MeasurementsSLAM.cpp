#include <doctest/doctest.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>

#include "../../src/GaussianInfo.hpp"
#include "../../src/SystemSLAM.h"
#include "../../src/SystemSLAMPoseLandmarks.h"
#include "../../src/MeasurementSLAMUniqueTagBundle.h"
#include "../../src/Camera.h"
#include "../../src/Pose.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision,0,", ",";\n","","","[","]")));
#endif

// Helper: load calibration used by the pipeline (keeps FOV/gating realistic).
static Camera loadTestCamera()
{
    Camera cam;
    std::filesystem::path cameraPath("test/data/camera.xml");
    REQUIRE(std::filesystem::exists(cameraPath));
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    REQUIRE(fs.isOpened());
    fs["camera"] >> cam;
    return cam;
}

// Helper: GaussianInfo for n = 12 (body) + 6·N (pose-landmarks). Here N=1.
static GaussianInfo<double> makeDensityWithOneTag(const Eigen::Vector3d& tag_pos_n)
{
    const int n = 12 + 6;
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
    mu.segment<3>(6).setZero();    // r_BN^n
    mu.segment<3>(9).setZero();    // Θ
    mu.segment<3>(12) = tag_pos_n; // landmark position
    mu.segment<3>(15).setZero();   // landmark orientation
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n,n);
    return GaussianInfo<double>::fromSqrtMoment(mu, S);
}

// Helper: 4 ArUco corners around pixel center (TL,TR,BR,BL). Matches bundle packing (2×4 per tag).
static Eigen::Matrix<double,2,4> makeCornersAroundCenter(const cv::Vec2d& c)
{
    const double h = 4.0;
    Eigen::Matrix<double,2,4> Y;
    Y.col(0) << c[0] - h, c[1] - h;
    Y.col(1) << c[0] + h, c[1] - h;
    Y.col(2) << c[0] + h, c[1] + h;
    Y.col(3) << c[0] - h, c[1] + h;
    return Y;
}

// -------------------------------------------------------------------------------------------------
// UniqueTagBundle (1 tag)
// Purpose:
//  • Validate dimensions of ∂ℓ/∂x and ∂²ℓ/∂x² returned by logLikelihood (size n).
//  • Validate first-order identity  (ℓ(x+εd)−ℓ(x))/ε ≈ (∂ℓ/∂x)ᵀ d  (assignment: measurement model,
//    linearization used in EKF/GN).
// -------------------------------------------------------------------------------------------------
SCENARIO("MeasurementSLAMUniqueTagBundle: log-likelihood API and dimensions")
{
    GIVEN("A system with one tag and a camera")
    {
        Camera cam = loadTestCamera();
        Eigen::Vector3d tag_n(0.2, -0.1, 5.0);
        auto density = makeDensityWithOneTag(tag_n);
        SystemSLAMPoseLandmarks sys(density);
        Eigen::VectorXd x = density.mean();

        // Project center with T_nb = I (cf. camera projection model in assignment).
        Posed Tnb;
        cv::Vec2d c = cam.worldToPixel(cv::Vec3d(tag_n.x(), tag_n.y(), tag_n.z()), Tnb);

        Eigen::Matrix<double,2,4> Y1 = makeCornersAroundCenter(c);
        Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, 4);
        Y.leftCols<4>() = Y1;

        std::vector<int> ids = {0};
        double t = 0.0;
        MeasurementSLAMUniqueTagBundle meas(t, Y, cam, ids);

        WHEN("logLikelihood(x,•) is evaluated")
        {
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            double l = meas.logLikelihood(x, sys, g, H);

            THEN("Gradient/Hessian sizes match n = x.size()")
            {
                CHECK(std::isfinite(l));
                REQUIRE(g.size() == x.size());
                REQUIRE(H.rows() == x.size());
                REQUIRE(H.cols() == x.size());
            }

            THEN("Directional derivative equals gᵀd (first-order check)")
            {
                // Finite-difference check of ∂ℓ/∂x along d (EKF/GN consistency).
                Eigen::VectorXd d = Eigen::VectorXd::Zero(x.size());
                d(6) = 1e-3;  // δr_x
                d(7) = -1e-3; // δr_y

                const double eps = 1e-2;
                const double fd  = (meas.logLikelihood(x + eps*d, sys) - l)/eps;
                const double dot = g.dot(d);
                CHECK(fd == doctest::Approx(dot).epsilon(1e-2));
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// UniqueTagBundle (2 tags)
// Purpose:
//  • Same first-order identity with multiple IDs and 2×(4·#tags) packing.
//  • Confirms ID↔landmark mapping and bundle assembly preserve ∂ℓ/∂x correctness.
// -------------------------------------------------------------------------------------------------
SCENARIO("MeasurementSLAMUniqueTagBundle: two tags — first-order consistency of log-likelihood gradient")
{
    GIVEN("A system with two pose-landmarks visible and a calibrated camera")
    {
        Camera cam = loadTestCamera();

        const double fx = cam.cameraMatrix.at<double>(0,0);
        const double fy = cam.cameraMatrix.at<double>(1,1);
        const double cx = cam.cameraMatrix.at<double>(0,2);
        const double cy = cam.cameraMatrix.at<double>(1,2);
        const double W  = cam.imageSize.width;
        const double H  = cam.imageSize.height;

        // Inverse pinhole at depth Z: X = (u−cx)/fx·Z , Y = (v−cy)/fy·Z  (assignment: camera model).
        auto pixelToWorldAtDepth = [&](double u, double v, double Z){
            return Eigen::Vector3d((u - cx)/fx * Z, (v - cy)/fy * Z, Z);
        };

        cv::Vec2d c0_px(0.55*W, 0.55*H);
        cv::Vec2d c1_px(0.70*W, 0.40*H);
        double Z0 = 5.0, Z1 = 6.0;
        Eigen::Vector3d tag0_n = pixelToWorldAtDepth(c0_px[0], c0_px[1], Z0);
        Eigen::Vector3d tag1_n = pixelToWorldAtDepth(c1_px[0], c1_px[1], Z1);

        // n = 12 + 2*6 (two pose landmarks).
        const int n = 12 + 2*6;
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
        mu.segment<3>(6).setZero();   // r_BN^n
        mu.segment<3>(9).setZero();   // Θ
        mu.segment<3>(12) = tag0_n;   // landmark 0
        mu.segment<3>(15).setZero();
        mu.segment<3>(18) = tag1_n;   // landmark 1
        mu.segment<3>(21).setZero();

        Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n,n);
        auto density = GaussianInfo<double>::fromSqrtMoment(mu, S);
        SystemSLAMPoseLandmarks sys(density);
        Eigen::VectorXd x = density.mean();

        auto corners = [&](const cv::Vec2d& c){
            const double h = 8.0;
            Eigen::Matrix<double,2,4> Y;
            Y.col(0) << c[0]-h, c[1]-h;
            Y.col(1) << c[0]+h, c[1]-h;
            Y.col(2) << c[0]+h, c[1]+h;
            Y.col(3) << c[0]-h, c[1]+h;
            return Y;
        };

        Eigen::Matrix<double,2,8> Y;
        Y.leftCols<4>()  = corners(c0_px);
        Y.rightCols<4>() = corners(c1_px);

        std::vector<int> ids = {0, 1};
        double t = 0.0;
        MeasurementSLAMUniqueTagBundle meas(t, Y, cam, ids);
        meas.setIdByLandmark({0, 1}); // landmark i ↔ tag id i

        WHEN("Comparing dℓ/dx vs finite differences along d")
        {
            Eigen::VectorXd g; Eigen::MatrixXd H;
            double l0 = meas.logLikelihood(x, sys, g, H);

            // d perturbs base pose only (map is static in process model).
            Eigen::VectorXd d = Eigen::VectorXd::Zero(n);
            d.segment<3>(6) <<  0.7, -0.3, 0.5;  // δr
            d.segment<3>(9) << -0.2,  0.4, 0.1;  // δΘ
            d.normalize();

            const double eps = 1e-3;
            const double fd   = (meas.logLikelihood(x + eps*d, sys) - l0)/eps;
            const double dotg = g.dot(d);
            CHECK(fd == doctest::Approx(dotg).epsilon(5e-2));
        }
    }
}
