#include <filesystem>
#include <string>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "SLAMProcessor.h"
#include "Plot.h"

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
    if (!camera.load(cameraPath)) {
        std::cout << "Error: Failed to load camera calibration from " << cameraPath << std::endl;
        return;
    }

    // Display loaded calibration data
    std::cout << "Camera calibration loaded from: " << cameraPath << std::endl;
    camera.printCalibration(); // This method should exist from Lab 4

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Processing video: " << videoPath.filename() << std::endl;
    std::cout << "Scenario " << scenario << " | Interactive " << interactive << " | Frames: " << nFrames << std::endl;

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize;
        frameSize.width     = 2*cap.get(cv::CAP_PROP_FRAME_WIDTH);  // Dual-pane layout
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double outputFps    = fps;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
        std::cout << "Exporting visualization to: " << outputPath << std::endl;
    }

    // Visual navigation

    // Initialisation
    SLAMProcessor slamProcessor(scenario, camera);
    //Plot plot(camera);
    
    int frameCount = 0;
    std::cout << "Starting SLAM processing..." << std::endl;

    while (true)
    {
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty())
        {
            break;
        }

        frameCount++;

        // Process frame
        slamProcessor.processFrame(imgin);

        // Update state - Get SLAM estimates
        auto detectedFeatures = slamProcessor.getDetectedFeatures();
        auto featureStatus = slamProcessor.getFeatureStatus();
        auto cameraPos = slamProcessor.getCameraPosition();
        auto cameraRot = slamProcessor.getCameraRotation();
        auto cameraCov = slamProcessor.getCameraCovariance();
        auto landmarkPos = slamProcessor.getLandmarkPositions();
        auto landmarkCov = slamProcessor.getLandmarkCovariances();
        auto landmarkStatus = slamProcessor.getLandmarkStatus();

        // Update plot
        //plot.updateImage(imgin, detectedFeatures, featureStatus);
        //plot.updateScene(cameraPos, cameraRot, cameraCov, landmarkPos, landmarkCov, landmarkStatus);
        //plot.render();

        // Handle interactivity based on assignment specs
        bool isLastFrame = (frameCount >= nFrames);
        if (interactive == 2) {
            // Interactive on all frames
            //plot.handleInteractivity(interactive, false);
        } else if (interactive == 1 && isLastFrame) {
            // Interactive only on last frame
            //plot.handleInteractivity(interactive, true);
        }

        // Progress feedback
        if (frameCount % 30 == 0 || isLastFrame) {
            std::cout << "Frame " << frameCount << "/" << nFrames 
                      << " | Features: " << detectedFeatures.size() 
                      << " | Camera pos: [" << cameraPos.transpose() << "]" << std::endl;
        }

        // Write output frame 
        if (doExport)
        {
            // cv::Mat imgout = plot.getFrame(); // Get the dual-pane visualization frame
            // if (!imgout.empty()) {
            //     bufferedVideoWriter.write(imgout);
            // }
        }
    }

    // Final cleanup and statistics
    std::cout << "SLAM processing complete:" << std::endl;
    std::cout << "  Total frames processed: " << frameCount << std::endl;
    std::cout << "  Final camera position: " << slamProcessor.getCameraPosition().transpose() << std::endl;
    std::cout << "  Total landmarks tracked: " << slamProcessor.getLandmarkPositions().size() << std::endl;

    if (doExport)
    {
         bufferedVideoWriter.stop();
         std::cout << "Video export completed: " << outputPath << std::endl;
    }
    bufferedVideoReader.stop();
}