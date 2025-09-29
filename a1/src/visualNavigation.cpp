#include <filesystem>
#include <string>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <array>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/aruco.hpp>

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

static SystemSLAMPoseLandmarks makeInitialPoseSystem()
{
    Eigen::VectorXd mu(12); mu.setZero();
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(12,12) * 1e-3;
    auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
    return SystemSLAMPoseLandmarks(p0);
}


void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario, int interactive, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());

    // Output video path
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // Load camera calibration
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

    // Display loaded calibration data
    camera.printCalibration();

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    // Ensure camera.imageSize is set (Plot uses it for aspect)
    {
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (camera.imageSize.width <= 0 || camera.imageSize.height <= 0)
            camera.imageSize = cv::Size(w, h);
    }

    // Create Plot now so we can query its 540-based render size
    Plot plot(camera);
    cv::Size plotSize = plot.renderSize();

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize = plotSize;
        double outputFps    = fps;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Visual navigation



    // Initialisation
    // Choose system by scenario
    std::unique_ptr<SystemSLAM> systemPtr;
    if (scenario == 1) systemPtr = std::make_unique<SystemSLAMPoseLandmarks>(makeInitialPoseSystem());
    else               systemPtr = std::make_unique<SystemSLAMPointLandmarks>(makeInitialPointSystem());
    SystemSLAM& system = *systemPtr;

    {
        // start plot with an empty measurement
        Eigen::Matrix<double,2,Eigen::Dynamic> Y0(2,0);
        MeasurementPointBundle m0(0.0, Y0, camera);
        plot.setData(system, m0);
    }

    int frameIdx = 0;
    while (true)
    {
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty())
        {
            break;
        }

        // Choose what to show on the left pane
        cv::Mat viewImg = imgin;

        double t = (fps > 0.0) ? (frameIdx / fps) : frameIdx;

        // Process frame
        if (scenario == 1)
        {
           // Detect using Lab-2 style function (new OpenCV backend)
            // Pick the dictionary that matches your printed tags:
            //   cv::aruco::DICT_6X6_250  or  cv::aruco::DICT_4X4_50
            ArucoDetections detsCV = detectArucoLab2(imgin, cv::aruco::DICT_6X6_250, /*refine*/true);

            // Build measurement Y (2 x 4*Ntags)
            Eigen::Matrix<double,2,Eigen::Dynamic> Y = buildYFromAruco(detsCV.corners);

            // Convert to our TagDetection format for the UniqueTag measurement
            std::vector<TagDetection> dets;
            dets.reserve(detsCV.ids.size());
            for (size_t k = 0; k < detsCV.ids.size(); ++k)
            {
                TagDetection td;
                td.id = detsCV.ids[k];
                for (int c = 0; c < 4; ++c)
                {
                    td.corners[c] = Eigen::Vector2d(detsCV.corners[k][c].x, detsCV.corners[k][c].y);
                }
                dets.push_back(td);
            }

            // Stage-1: create measurement (no fusion), show annotated image
            MeasurementSLAMUniqueTagBundle meas(t, Y, camera, dets, /*tagSizeMeters=*/0.16);

            // Left pane will show what we put in system.view()
            system.view() = detsCV.annotated.empty() ? imgin : detsCV.annotated;

            plot.setData(system, meas);
            plot.render();
            if (doExport) bufferedVideoWriter.write(plot.getFrame());
        }
        else
        {
            // Your existing Scenario 2/3 handling
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
