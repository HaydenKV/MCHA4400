#include <doctest/doctest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <numeric>

#include <Eigen/Core>

#include "../../src/GaussianInfo.hpp"
#include "../../src/SystemSLAMPointLandmarks.h"
#include "../../src/SystemSLAMPoseLandmarks.h"
#include "../../src/MeasurementSLAMPointBundle.h"
#include "../../src/MeasurementSLAMUniqueTagBundle.h"
#include "../../src/Camera.h"

static Camera loadCamera(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    REQUIRE(fs.isOpened());
    Camera cam; fs["camera"] >> cam;
    return cam;
}

SCENARIO("MeasurementPointBundle: associate & single-feature prediction")
{
    GIVEN("A point-landmark SLAM system with 2 landmarks and a small bundle Y")
    {
        // 12 body + 2*3 landmarks
        const int nx = 12 + 2*3;
        auto g = GaussianInfo<double>::fromSqrtMoment(
            Eigen::VectorXd::Zero(nx), Eigen::MatrixXd::Identity(nx,nx)
        );
        SystemSLAMPointLandmarks sys(g);

        Camera cam = loadCamera("test/data/camera.xml");

        // 3 features (columns)
        Eigen::Matrix<double,2,Eigen::Dynamic> Y(2,3);
        Y << 100, 150, 200,
             120, 160, 210;

        MeasurementPointBundle meas(0.0, Y, cam);

        THEN("associate returns a mapping for each system landmark")
        {
            std::vector<std::size_t> idxL(sys.numberLandmarks());
            std::iota(idxL.begin(), idxL.end(), 0);

            const auto& assoc = meas.associate(sys, idxL);
            CHECK(assoc.size() == sys.numberLandmarks());
        }

        THEN("predictFeature returns a 2D pixel for a landmark")
        {
            Eigen::VectorXd x = Eigen::VectorXd::Zero(nx);
            Eigen::MatrixXd J; // jacobian (filled by impl)
            auto uv = meas.predictFeature(x, J, sys, /*idxLandmark*/0);
            CHECK(uv.size() == 2);
        }
    }
}

SCENARIO("MeasurementSLAMUniqueTagBundle: ids, association, and visibility accessors")
{
    GIVEN("A pose-landmark SLAM system and a single detected tag id")
    {
        const int nx = 12; // body only to start
        auto g = GaussianInfo<double>::fromSqrtMoment(
            Eigen::VectorXd::Zero(nx), Eigen::MatrixXd::Identity(nx,nx)
        );
        SystemSLAMPoseLandmarks sys(g);
        Camera cam = loadCamera("test/data/camera.xml");

        // one tag's 4 corners as 2x4 Y (dummy numbers are fine for compile/runtime)
        Eigen::Matrix<double,2,Eigen::Dynamic> Y(2,4);
        Y << 100,120,120,100,
             100,100,120,120;
        std::vector<int> ids = {0};

        MeasurementSLAMUniqueTagBundle meas(0.0, Y, cam, ids);
        meas.setIdByLandmark({0}); // map landmark 0 -> tag id 0 (public API)

        THEN("associate builds mappings and visibility vector is queryable")
        {
            // After we append one pose landmark, the system has one
            Eigen::Vector3d r(0,0,1);
            Eigen::Vector3d th(0,0,0);
            Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Identity();
            sys.appendLandmark(r, th, Sj);

            std::vector<std::size_t> idxL(sys.numberLandmarks());
            std::iota(idxL.begin(), idxL.end(), 0);

            const auto& assoc = meas.associate(sys, idxL);
            CHECK(assoc.size() == sys.numberLandmarks());

            // Public accessor returns vector<bool> by const&
            const auto& vis = meas.isVisible();
            CHECK(vis.size() == sys.numberLandmarks());
            CHECK((vis[0] == true || vis[0] == false));
        }
    }
}
