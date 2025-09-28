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
#include "MeasurementSLAMPointBundle.h"
#include "GaussianInfo.hpp"

#include "imagefeatures.h"

static SystemSLAMPointLandmarks makeInitialSystem()
{
    // State: [vBNb(3); omegaBNb(3); rBNn(3); Thetanb(3); landmarks(3*4)]
    Eigen::VectorXd mu(24);
    mu.setZero();
    mu.segment<3>(12) << 0.0, 0.0, 0.0;   // L1
    mu.segment<3>(15) << 1.0, 0.0, 0.0;   // L2
    mu.segment<3>(18) << 1.0, 1.0, 0.0;   // L3
    mu.segment<3>(21) << 0.0, 1.0, 0.0;   // L4
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(24,24) * 1e-3;
    auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
    return SystemSLAMPointLandmarks(p0);
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
    SystemSLAMPointLandmarks system = makeInitialSystem();

    {
        // Start with an empty measurement; we’ll rebuild per frame
        Eigen::Matrix<double,2,Eigen::Dynamic> Y0(2,0);
        MeasurementPointBundle m0(0.0, Y0, camera);
        // Plot initial state
        plot.setData(system, m0);
    }

    // Scenario 1 bookkeeping: tagID -> 4 landmark indices (corner order TL,TR,BR,BL)
    std::unordered_map<int, std::array<int,4>> tagIdToLmks;

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
        // TODO: detect features for 'scenario' and fill Y as a 2xM matrix
        // Y = <detector output>;
        // measurement = MeasurementPointBundle(/*time*/, Y, camera);
        // Scenario switch
        if (scenario == 1) {
            // Use Lab-2 detector exactly as-is (draws boxes + IDs on the image)
            viewImg = detectAndDrawArUco(imgin, /*maxNumFeatures ignored*/ 0);
        }


        // Update state
        // TODO: when ready, call measurement.update(system) to apply measurement update
        system.view() = imgin;

        // Build the measurement for THIS frame (temporary object)
        Eigen::Matrix<double,2,Eigen::Dynamic> Ynow(2,0);
        // TODO: when wiring tags, fill Ynow with the detected corner pixels (2 x M)

        MeasurementPointBundle meas(t, Ynow, camera);
        // TODO (later): meas.update(system);  // when we enable fusion

        // Update plot using this frame's measurement
        plot.setData(system, meas);
        plot.render();

        // Write output frame 
        if (doExport)
        {
            cv::Mat imgout = plot.getFrame(); // TODO: Uncomment this to get the frame image
            bufferedVideoWriter.write(imgout);
        }

        // Optional interactivity behaviour (kept outside the template blocks to avoid edits)
        if (interactive == 2 || (interactive == 1 && (--nFrames == 0)))
            plot.start();

        ++frameIdx;
    }

    if (doExport) bufferedVideoWriter.stop();
    bufferedVideoReader.stop();
}
