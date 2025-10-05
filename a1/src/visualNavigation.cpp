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
#include "Camera.h"
#include "Pose.hpp"
#include "Plot.h"
#include "SystemSLAMPointLandmarks.h"
#include "SystemSLAMPoseLandmarks.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "GaussianInfo.hpp"
#include "imagefeatures.h"
#include "rotation.hpp"

// ============================================================================
// SCENARIO 1 CONSTANTS (Shared across the file)
// ============================================================================
namespace {
    constexpr float  TAG_SIZE_METERS = 0.166f;       // ArUco tag edge length (166mm)
    constexpr double REPROJ_ERR_THRESH_PX = 4.0;     // IPPE reprojection gate (pixels)
    
    // Initial landmark uncertainty (moderate values for robustness)
    constexpr double INIT_POS_SIGMA = 0.4;           // ~40 cm position uncertainty
    constexpr double INIT_ANG_SIGMA = 20.0 * M_PI / 180.0;  // ~20° orientation uncertainty
}

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

void runVisualNavigationFromVideo(
    const std::filesystem::path& videoPath,
    const std::filesystem::path& cameraPath,
    int scenario,
    int interactive,
    const std::filesystem::path& outputDirectory,
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
    
    // Main loop
    int frameIdx = 0;
    while (true)
    {
        // ArUco detection
        // Landmark initialization
        // Build measurement
        // Time update (via Event::process)
        // Measurement update (via Event::process)
        // Visualization

        // std::cout << "Frame " << frameIdx << " - Start" << std::endl;

        if (max_frames > 0 && frameIdx >= max_frames) break;
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty()) break;

        const double t = (fps > 0.0) ? (frameIdx / fps) : frameIdx;

        if (scenario == 1)
        {
            // Detect + pose (IPPE) + reprojection gating
            std::vector<cv::Vec3d> rvecs, tvecs;
            std::vector<double> meanErrs;
            ArucoDetections dets = detectArUcoPOSE(
                imgin,
                cv::aruco::DICT_6X6_250,
                /*doCornerRefine*/ true,
                camera.cameraMatrix,
                camera.distCoeffs,
                TAG_SIZE_METERS,
                &rvecs, &tvecs, &meanErrs,
                REPROJ_ERR_THRESH_PX,
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

            // Initialize new landmarks for any unseen tag IDs
            for (std::size_t i = 0; i < dets.ids.size(); ++i)
            {
                const int tagId = dets.ids[i];
                if (id2lm.find(tagId) != id2lm.end())
                    continue; // Already known

                // Skip if detection too close to image border
                if (!camera.areCornersInside(dets.corners[i]))
                    continue;

                // Skip the tag CENTER ray is inside FOV after distortion (with the same margin)
                const cv::Vec3d rCj(tvecs[i][0], tvecs[i][1], tvecs[i][2]); // camera->tag
                if (!camera.isVectorWithinFOVConservative(rCj))
                    continue;

                // Convert IPPE pose (camera→tag) to world coordinates
                const Eigen::Matrix3d Rcj = rodriguesToRot(rvecs[i]);
                const Eigen::Vector3d rCj(tvecs[i][0], tvecs[i][1], tvecs[i][2]);
                
                const Eigen::Matrix3d Rnj = Rnc * Rcj;
                const Eigen::Vector3d rnj = rCNn + Rnc * Eigen::Vector3d(rCj[0], rCj[1], rCj[2]);
                const Eigen::Vector3d Thetanj = rot2rpy(Rnj);

                // Initial uncertainty (moderate for robustness)
                Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Zero();
                Sj(0,0) = INIT_POS_SIGMA;  Sj(1,1) = INIT_POS_SIGMA;  Sj(2,2) = INIT_POS_SIGMA;
                Sj(3,3) = INIT_ANG_SIGMA;  Sj(4,4) = INIT_ANG_SIGMA;  Sj(5,5) = INIT_ANG_SIGMA;

                const std::size_t j = sysPose->appendLandmark(rnj, Thetanj, Sj);
                if (j >= id_by_landmark.size()) id_by_landmark.resize(j+1, -1);
                id_by_landmark[j] = tagId;
                id2lm[tagId] = j;
            }

            // Build measurement matrix Y (2 × 4N) with OpenCV order (TL,TR,BR,BL)
            const std::size_t N = dets.ids.size();
            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, 4*N);
            for (std::size_t i = 0; i < N; ++i) {
                const auto& c = dets.corners[i];
                for (int k = 0; k < 4; ++k) {
                    Y(0, 4*i + k) = c[k].x;
                    Y(1, 4*i + k) = c[k].y;
                }
            }

            // GUARD: Verify Y packing is correct (4 columns per detection)
            assert(Y.cols() == 4 * static_cast<int>(N) && 
                   "Y packing error: should have 4 columns per tag detection");

            // Display
            system.view() = dets.annotated.empty() ? imgin : dets.annotated;

            // Create measurement event
            MeasurementSLAMUniqueTagBundle meas(t, Y, camera, dets.ids);
            meas.setIdByLandmark(id_by_landmark);

            // Process (time update + measurement update)
            meas.process(system);

            // Update persistent state
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


            // Visualization
            plot.setData(system, meas);
        }
        // ... [Other scenarios] ...
        else
        {
            // Other scenarios unchanged
            system.view() = imgin;
            Eigen::Matrix<double,2,Eigen::Dynamic> Ynow(2,0);
            MeasurementPointBundle meas(t, Ynow, camera);
            plot.setData(system, meas);
        }

        plot.render();
        if (doExport) bufferedVideoWriter.write(plot.getFrame());

        if (interactive == 2 || (interactive == 1 && (--nFrames == 0)))
            plot.start();

        frameIdx++;
    }
    if (doExport) bufferedVideoWriter.stop();
    bufferedVideoReader.stop();
}