#include <doctest/doctest.h>
#include <Eigen/Core>

#include "../../src/GaussianInfo.hpp"
#include "../../src/SystemSLAMPointLandmarks.h"
#include "../../src/SystemSLAMPoseLandmarks.h"
#include "../../src/Camera.h"
#include "../../src/Pose.hpp"

SCENARIO("SystemSLAMPointLandmarks indexing and counts")
{
    GIVEN("A Gaussian density with body + 2 point landmarks (3D each)")
    {
        const int nx = 12 + 2*3;
        auto g = GaussianInfo<double>::fromSqrtMoment(
            Eigen::VectorXd::Zero(nx), Eigen::MatrixXd::Identity(nx,nx)
        );
        SystemSLAMPointLandmarks sys(g);

        THEN("numberLandmarks and position index are correct")
        {
            CHECK(sys.numberLandmarks() == 2);
            CHECK(sys.landmarkPositionIndex(0) == 12);
            CHECK(sys.landmarkPositionIndex(1) == 15);
        }
    }
}

SCENARIO("SystemSLAMPoseLandmarks indexing and appendLandmark")
{
    GIVEN("An empty pose-landmark SLAM system")
    {
        const int nx = 12; // body only
        auto g = GaussianInfo<double>::fromSqrtMoment(
            Eigen::VectorXd::Zero(nx), Eigen::MatrixXd::Identity(nx,nx)
        );
        SystemSLAMPoseLandmarks sys(g);

        THEN("Appending a 6-DOF pose landmark increases size and index as expected")
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

SCENARIO("Camera FOV and world-to-pixel logic sanity")
{
    GIVEN("A calibrated camera loaded from XML")
    {
        cv::FileStorage fs("test/data/camera.xml", cv::FileStorage::READ);
        REQUIRE(fs.isOpened());
        Camera cam; fs["camera"] >> cam;

        Posed Tnb; // identity
        cv::Vec3d front(0,0,2.0);
        cv::Vec3d back (0,0,-2.0);

        THEN("Points in front project to finite pixels")
        {
            auto uv = cam.worldToPixel(front, Tnb);
            CHECK(std::isfinite(uv[0]));
            CHECK(std::isfinite(uv[1]));
        }

        THEN("Points behind are outside FOV")
        {
            CHECK_FALSE(cam.isWorldWithinFOV(back, Tnb));
        }
    }
}
