#include <filesystem>
#include <string>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <array>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

#include "BufferedVideo.h"
#include "visualNavigation.h"

// === Assignment code you already have ===
#include "Camera.h"
#include "Pose.hpp"

// === Lab 8 visualisation scaffold ===
#include "Plot.h"
#include "SystemSLAMPointLandmarks.h"
#include "SystemSLAMPoseLandmarks.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "GaussianInfo.hpp"

#include "imagefeatures.h"
#include "rotation.hpp"

// Helper: convert Rodrigues rvec to rotation matrix
static inline Eigen::Matrix3d rodriguesToRot(const cv::Vec3d& rvec)
{
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    Eigen::Matrix3d R;
    cv::cv2eigen(Rcv, R);
    return R;
}

// Helper: ensure a matrix is upper-triangular (safeguard for Sj)
static inline Eigen::Matrix<double,6,6> makeUpper(const Eigen::Matrix<double,6,6>& M)
{
    Eigen::Matrix<double,6,6> U = M;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < r; ++c)
            U(r,c) = 0.0;
    return U;
}

// helper: build an empty pose-landmark system (12-state body only)
static SystemSLAMPointLandmarks makeInitialPointSystem()
{
    // From your previous working version (4 seeded points)
    Eigen::VectorXd mu(24);
    mu.setZero();
    mu.segment<3>(12) << 0.0, 0.0, 0.0;
    mu.segment<3>(15) << 1.0, 0.0, 0.0;
    mu.segment<3>(18) << 1.0, 1.0, 0.0;
    mu.segment<3>(21) << 0.0, 1.0, 0.0;
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(24,24) * 1e-3;
    auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
    return SystemSLAMPointLandmarks(p0);
}

// helper: build a pose-landmark system with ONE pose landmark (12 body + 6 landmark = 18)
static SystemSLAMPoseLandmarks makeInitialPoseSystem()
{
    // Body (12) + one pose-landmark (6) = 18-dim state
    const int nBody = 12;
    const int nLm   = 6;

    Eigen::VectorXd mu(nBody + nLm);
    mu.setZero();

    // Seed the landmark ~1m in front of the camera, no rotation
    mu.segment<3>(nBody + 0) << 0.0, 0.0, 1.0;   // r_n^L/N
    mu.segment<3>(nBody + 3).setZero();          // Theta_nL (Euler)

    // Build sqrt-covariance S (NOT covariance) with realistic sigmas
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nBody + nLm, nBody + nLm);

    // Body uncertainty (keep tight so the camera quadric isn’t huge)
    S.block(0, 0, nBody, nBody).setIdentity();
    S.block(0, 0, nBody, nBody) *= 1e-2;         // sqrt-cov ≈ 0.01 on body state

    // Landmark position: ~0.25 m σ  → sqrt-cov = 0.25
    S.block(nBody + 0, nBody + 0, 3, 3).setIdentity();
    S.block(nBody + 0, nBody + 0, 3, 3) *= 0.25;

    // Landmark orientation: ~15° σ → sqrt-cov = 15° in radians
    const double sigmaEuler = 15.0 * M_PI / 180.0; // ≈ 0.262 rad
    S.block(nBody + 3, nBody + 3, 3, 3).setIdentity();
    S.block(nBody + 3, nBody + 3, 3, 3) *= sigmaEuler;

    auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
    return SystemSLAMPoseLandmarks(p0);
}

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath,
                                  const std::filesystem::path & cameraPath,
                                  int scenario, int interactive,
                                  const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());

    // ------------------ Output setup ------------------
    std::filesystem::path outputPath;
    const bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // ------------------ Load camera -------------------
    Camera camera;
    {
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cerr << "File: " << cameraPath << " does not exist or cannot be opened\n";
            std::exit(EXIT_FAILURE);
        }
        fs["camera"] >> camera;
        fs.release();
    }
    camera.printCalibration();

    // ------------------ Open video --------------------
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    // Ensure camera.imageSize is set (Plot uses it)
    {
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (camera.imageSize.width <= 0 || camera.imageSize.height <= 0)
            camera.imageSize = cv::Size(w, h);
    }

    // ------------------ Plot & export -----------------
    Plot plot(camera);
    const cv::Size plotSize = plot.renderSize();

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        const int codec = cv::VideoWriter::fourcc('m','p','4','v');
        videoOut.open(outputPath.string(), codec, fps, plotSize);
        bufferedVideoWriter.start(videoOut);
    }

    // ------------------ Build system ------------------
    // Scenario 1 uses a pose-landmark system starting with ONLY the 12 body states (no landmarks yet).
    std::unique_ptr<SystemSLAM> systemPtr;
    if (scenario == 1)
    {
        Eigen::VectorXd mu_body(12); mu_body.setZero();
        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12) * 1e-2; // tight prior on body
        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPoseLandmarks>(SystemSLAMPoseLandmarks(p0));
    }
    else
    {
        systemPtr = std::make_unique<SystemSLAMPointLandmarks>(makeInitialPointSystem());
    }
    SystemSLAM& system = *systemPtr;

    // Prime plot with an empty measurement
    {
        Eigen::Matrix<double,2,Eigen::Dynamic> Y0(2,0);
        MeasurementPointBundle m0(0.0, Y0, camera);
        plot.setData(system, m0);
    }

    // Persistent mapping: landmark index -> tag ID
    static std::vector<int> id_by_landmark;
    id_by_landmark.clear();

    const float tagSizeMeters = 0.166f; // set to your printed tag edge length

    // Object points for a square tag in its own frame, TL,TR,BR,BL
    std::vector<cv::Point3f> objPts{
        {-tagSizeMeters/2.f,  tagSizeMeters/2.f, 0.f}, // TL
        { tagSizeMeters/2.f,  tagSizeMeters/2.f, 0.f}, // TR
        { tagSizeMeters/2.f, -tagSizeMeters/2.f, 0.f}, // BR
        {-tagSizeMeters/2.f, -tagSizeMeters/2.f, 0.f}  // BL
    };

    // Helper: reorder detected corners to TL,TR,BR,BL (like OpenCV samples)
    auto reorderTLTRBRBL = [](const std::array<cv::Point2f,4>& in)->std::vector<cv::Point2f>
    {
        std::vector<cv::Point2f> pts(in.begin(), in.end());
        // sort by y to split top vs bottom
        std::sort(pts.begin(), pts.end(),
                  [](const cv::Point2f& a, const cv::Point2f& b){ return a.y < b.y; });
        std::vector<cv::Point2f> top(pts.begin(), pts.begin()+2);
        std::vector<cv::Point2f> bot(pts.begin()+2, pts.end());
        std::sort(top.begin(), top.end(),
                  [](const cv::Point2f& a, const cv::Point2f& b){ return a.x < b.x; }); // TL,TR
        std::sort(bot.begin(), bot.end(),
                  [](const cv::Point2f& a, const cv::Point2f& b){ return a.x < b.x; }); // BL,BR
        // Return TL,TR,BR,BL
        return { top[0], top[1], bot[1], bot[0] };
    };

    int frameIdx = 0;
    while (true)
    {
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty()) break;

        const double t = (fps > 0.0) ? (frameIdx / fps) : frameIdx;

        if (scenario == 1)
        {
            // 1) Detect ArUco tags
            ArucoDetections dets = detectArucoLab2(imgin, cv::aruco::DICT_6X6_250, /*refine*/true);

            // 2) Per-marker PnP using SOLVEPNP_IPPE_SQUARE with corners reordered to TL,TR,BR,BL
            std::vector<cv::Vec3d> rvecs, tvecs;
            rvecs.reserve(dets.corners.size());
            tvecs.reserve(dets.corners.size());
            for (const auto& arr : dets.corners)
            {
                const std::vector<cv::Point2f> imgPts = reorderTLTRBRBL(arr);
                cv::Vec3d rvec, tvec;
                bool ok = cv::solvePnP(objPts, imgPts,
                                       camera.cameraMatrix, camera.distCoeffs,
                                       rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
                if (!ok) { rvec = cv::Vec3d(0,0,0); tvec = cv::Vec3d(0,0,0); }
                rvecs.push_back(rvec);
                tvecs.push_back(tvec);
            }

            // 3) Spawn new landmarks for any newly seen tag IDs (transform PnP pose to world frame)
            auto* sysPose = dynamic_cast<SystemSLAMPoseLandmarks*>(&system);
            assert(sysPose && "Scenario 1 expects SystemSLAMPoseLandmarks");

            // Ensure mapping length matches current number of landmarks
            if (id_by_landmark.size() < system.numberLandmarks())
                id_by_landmark.resize(system.numberLandmarks(), -1);

            // Current camera pose from the (mean) state
            const Eigen::VectorXd xmean = sysPose->density.mean();
            const Eigen::Vector3d rCNn  = SystemSLAM::cameraPosition(camera, xmean);
            const Eigen::Matrix3d Rnc   = SystemSLAM::cameraOrientation(camera, xmean);

            for (std::size_t i = 0; i < dets.ids.size(); ++i)
            {
                const int tagId = dets.ids[i];
                const bool known = std::find(id_by_landmark.begin(), id_by_landmark.end(), tagId) != id_by_landmark.end();
                if (known) continue;

                // Tag pose in camera frame (from our IPPE solve)
                cv::Mat Rcv;
                cv::Rodrigues(rvecs[i], Rcv); // 3x3, CV_64F
                Eigen::Matrix3d Rcj;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        Rcj(r,c) = Rcv.at<double>(r,c);
                const Eigen::Vector3d rCj(tvecs[i][0], tvecs[i][1], tvecs[i][2]);

                // Transform to nav/world frame
                const Eigen::Matrix3d Rnj = Rnc * Rcj;
                const Eigen::Vector3d rnj = rCNn + Rnc * rCj;

                // CHANGED LINE: Extract Euler angles from rotation matrix
                const Eigen::Vector3d Thetanj = rot2rpy(Rnj);  // Changed from Eigen::Vector3d::Zero()

                // Initial sqrt-cov (upper-triangular): 0.25 m on pos, 15° on eulers
                const double d2r = M_PI / 180.0;
                Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Zero();
                Sj.diagonal() << 0.25, 0.25, 0.25, (15.0*d2r), (15.0*d2r), (15.0*d2r);

                const std::size_t j = sysPose->appendLandmark(rnj, Thetanj, Sj);

                if (j >= id_by_landmark.size()) id_by_landmark.resize(j+1, -1);
                id_by_landmark[j] = tagId;
            }

            // 4) Build centroid measurement Yc (2 x N) + ids
            const std::size_t N = dets.ids.size();
            Eigen::Matrix<double,2,Eigen::Dynamic> Yc(2, static_cast<int>(N));
            for (std::size_t i = 0; i < N; ++i)
            {
                const auto & cs = dets.corners[i];
                double u=0.0, v=0.0;
                for (int c = 0; c < 4; ++c) { u += cs[c].x; v += cs[c].y; }
                Yc(0, static_cast<int>(i)) = u * 0.25;
                Yc(1, static_cast<int>(i)) = v * 0.25;
            }

            // 5) Compose left-pane overlay: green boxes (already drawn by detectArucoLab2) + RGB pose axes
            cv::Mat overlay = dets.annotated.empty() ? imgin.clone() : dets.annotated.clone();
            for (std::size_t i = 0; i < dets.ids.size(); ++i)
            {
                cv::drawFrameAxes(overlay,
                                  camera.cameraMatrix, camera.distCoeffs,
                                  rvecs[i], tvecs[i],
                                  tagSizeMeters * 1.5f, 2);
            }
            system.view() = overlay;

            // Predict system state forward
            system.predict(t);

            // 6) Fuse: ID-based association + optimiser update
            MeasurementSLAMUniqueTagBundle meas(t, Yc, camera, dets.ids);
            meas.setIdByLandmark(id_by_landmark);
            meas.process(system); // predict → associate(by ID) → update

            // 7) Plot both panes
            plot.setData(system, meas);
            plot.render();
            if (doExport) bufferedVideoWriter.write(plot.getFrame());
        }
        else
        {
            // Scenario 2/3: keep your existing point-bundle no-op view
            system.view() = imgin;
            Eigen::Matrix<double,2,Eigen::Dynamic> Ynow(2,0);
            MeasurementPointBundle meas(t, Ynow, camera);
            plot.setData(system, meas);
            plot.render();
            if (doExport) bufferedVideoWriter.write(plot.getFrame());
        }

        if (interactive == 2 || (interactive == 1 && (--nFrames == 0)))
            plot.start();

        ++frameIdx;
    }

    if (doExport) bufferedVideoWriter.stop();
    bufferedVideoReader.stop();
}