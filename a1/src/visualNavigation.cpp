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
#include <opencv2/imgproc.hpp>

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
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r,c) = Rcv.at<double>(r,c);
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

// Optional helper: Eigen::Vector3d -> cv::Vec3d
static inline cv::Vec3d toCv(const Eigen::Vector3d& v) { return cv::Vec3d(v.x(), v.y(), v.z()); }

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath,
                                  const std::filesystem::path & cameraPath,
                                  int scenario, int interactive,
                                  const std::filesystem::path & outputDirectory,
                                  int max_frames)
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

    std::cout << "Total frames in video: " << nFrames << std::endl;
    std::cout << "Video duration (approx): " << (nFrames / fps) << " seconds" << std::endl;

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
    std::unique_ptr<SystemSLAM> systemPtr;
    if (scenario == 1)
    {
        Eigen::VectorXd mu_body(12);
        mu_body.setZero(); // Start at origin, zero velocity

        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12); // tight prior on body
        S_body.block<6,6>(0,0) *= 1e-2;               // velocity
        const double d2r = M_PI / 180.0;
        S_body.block<3,3>(6,6) *= 1e-2;               // Position: 10cm
        S_body.block<3,3>(9,9) *= (1.0 * d2r);        // Orientation: 5°

        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPoseLandmarks>(SystemSLAMPoseLandmarks(p0));
    }
    else
    {
        Eigen::VectorXd mu(24);
        mu.setZero();
        mu.segment<3>(12) << 0.0, 0.0, 0.0;
        mu.segment<3>(15) << 1.0, 0.0, 0.0;
        mu.segment<3>(18) << 1.0, 1.0, 0.0;
        mu.segment<3>(21) << 0.0, 1.0, 0.0;
        Eigen::MatrixXd S = Eigen::MatrixXd::Identity(24,24) * 1e-3;
        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
        systemPtr = std::make_unique<SystemSLAMPointLandmarks>(SystemSLAMPointLandmarks(p0));
    }
    SystemSLAM& system = *systemPtr;

    // Prime plot with an empty measurement
    {
        Eigen::Matrix<double,2,Eigen::Dynamic> Y0(2,0);
        MeasurementPointBundle m0(0.0, Y0, camera);
        plot.setData(system, m0);
    }

    // Persistent mapping structures
    static std::vector<int> id_by_landmark;                 // landmark idx -> tag id
    static std::unordered_map<int, std::size_t> id2lm;      // tag id -> landmark idx
    id_by_landmark.clear();
    id2lm.clear();

    const float tagSizeMeters = 0.166f; // your printed tag edge length

    // ------------------ Main loop ---------------------
    int frameIdx = 0;
    while (true)
    {
        if (max_frames > 0 && frameIdx >= max_frames) break;  // <- early-out
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty()) break;

        // std::cout << "Frame " << frameIdx << " - Start" << std::endl;

        const double t = (fps > 0.0) ? (frameIdx / fps) : frameIdx;

        if (scenario == 1)
        {
            // Detect + pose (IPPE) + light gating in the detector itself
            std::vector<cv::Vec3d> rvecs, tvecs;
            std::vector<double> meanErrs;
            ArucoDetections dets = detectArUcoPOSE(
                imgin,
                cv::aruco::DICT_6X6_250,
                /*doCornerRefine*/ true,
                camera.cameraMatrix,
                camera.distCoeffs,
                tagSizeMeters,
                &rvecs, &tvecs, &meanErrs,
                /*reprojErrThreshPx*/ 6.0,   // slightly more permissive
                /*drawRejected*/ false
            );

            auto* sysPose = dynamic_cast<SystemSLAMPoseLandmarks*>(&system);
            assert(sysPose && "Scenario 1 expects SystemSLAMPoseLandmarks");

            if (id_by_landmark.size() < system.numberLandmarks())
                id_by_landmark.resize(system.numberLandmarks(), -1);

            // Current camera pose (mean)
            const Eigen::VectorXd xmean = sysPose->density.mean();
            const Eigen::Vector3d rCNn  = SystemSLAM::cameraPosition(camera, xmean);
            const Eigen::Matrix3d Rnc   = SystemSLAM::cameraOrientation(camera, xmean);

            // --- Conservative image-border gating for NEW landmark init ---
            const int W = camera.imageSize.width;
            const int H = camera.imageSize.height;
            const int BORDER_MARGIN = 15; // ~10–15 px as discussed

            auto nearBorder = [&](const std::array<cv::Point2f,4>& c)->bool {
                for (int k = 0; k < 4; ++k) {
                    const float u = c[k].x, v = c[k].y;
                    if (u < BORDER_MARGIN || u > (W-1-BORDER_MARGIN) ||
                        v < BORDER_MARGIN || v > (H-1-BORDER_MARGIN)) {
                        return true;
                    }
                }
                return false;
            };

            // Initialize new landmarks for any unseen tag IDs (ID → landmark)
            for (std::size_t i = 0; i < dets.ids.size(); ++i)
            {
                const int tagId = dets.ids[i];
                if (id2lm.find(tagId) != id2lm.end())
                    continue; // already known

                // Skip if detection too close to the image border (unreliable geometry)
                if (nearBorder(dets.corners[i]))
                    continue;

                // Camera->Marker (from IPPE), then World<-Marker
                cv::Mat Rcv;
                cv::Rodrigues(rvecs[i], Rcv);
                Eigen::Matrix3d Rcj;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        Rcj(r,c) = Rcv.at<double>(r,c);

                const Eigen::Vector3d rCj(tvecs[i][0], tvecs[i][1], tvecs[i][2]);

                const Eigen::Matrix3d Rnj = Rnc * Rcj;
                const Eigen::Vector3d rnj = rCNn + Rnc * rCj;
                const Eigen::Vector3d Thetanj = rot2rpy(Rnj);

                // Moderate init sqrt-covariance (position meters, Euler radians)
                Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Zero();
                // 0.1 5° Small Fast but may lock onto wrong solution
                // 0.4 20° Medium Moderate
                // 1.0 45° Large Slow
                const double pos_sigma = 0.15;                 // ~40 cm
                const double ang_sigma = 5.0 * M_PI/180.0;    // ~20°
                Sj(0,0) = pos_sigma;  Sj(1,1) = pos_sigma;  Sj(2,2) = pos_sigma;
                Sj(3,3) = ang_sigma;  Sj(4,4) = ang_sigma;  Sj(5,5) = ang_sigma;

                const std::size_t j = sysPose->appendLandmark(rnj, Thetanj, Sj);
                if (j >= id_by_landmark.size()) id_by_landmark.resize(j+1, -1);
                id_by_landmark[j] = tagId;
                id2lm[tagId] = j;
            }

            // Build measurement matrix Y (2 x 4N) with OpenCV order (TL,TR,BR,BL)
            const std::size_t N = dets.ids.size();
            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, 4*N);
            for (std::size_t i = 0; i < N; ++i) {
                const auto& c = dets.corners[i];
                for (int k = 0; k < 4; ++k) {
                    Y(0, 4*i + k) = c[k].x;
                    Y(1, 4*i + k) = c[k].y;
                }
            }

            // std::cout << "Frame " << frameIdx << " - After measurement" << std::endl;

            // Display
            system.view() = dets.annotated.empty() ? imgin.clone() : dets.annotated.clone();

            // NOTE: prediction is done inside Event::process() (Measurement::process)
            MeasurementSLAMUniqueTagBundle meas(t, Y, camera, dets.ids);
            meas.setIdByLandmark(id_by_landmark);
            // (we intentionally do NOT do any pruning or grace-based coloring)
            meas.process(system);
            // std::cout << "Frame " << frameIdx << " - After update" << std::endl;
            id_by_landmark = meas.idByLandmark();

            // Diagnostics
            if (frameIdx % 10 == 0) {
                std::cout << "\n=== Frame " << frameIdx << " ===\n";
                std::cout << "  Landmarks: " << system.numberLandmarks() << "\n";
                std::cout << "  Detected tags: " << dets.ids.size() << "\n";
                int nAssoc = 0;
                for (int idx : meas.idxFeatures()) if (idx >= 0) ++nAssoc;
                std::cout << "  Associated: " << nAssoc << "\n";
                const Eigen::VectorXd x = system.density.mean();
                const Eigen::Vector3d camPos = SystemSLAM::cameraPosition(camera, x);
                std::cout << "  Camera pos: [" << camPos.transpose() << "]\n";

                for (std::size_t i = 0; i < system.numberLandmarks(); ++i) {
                const auto posDen = system.landmarkPositionDensity(i);
                const Eigen::Vector3d std_dev = posDen.sqrtCov().diagonal();
                std::cout << "  Landmark " << i << " std: [" 
                        << std_dev.transpose() << "] m\n";
                }
            }

            // Render / export
            plot.setData(system, meas);
            plot.render();
            if (doExport) bufferedVideoWriter.write(plot.getFrame());
        }
        else
        {
            // Other scenarios unchanged
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

