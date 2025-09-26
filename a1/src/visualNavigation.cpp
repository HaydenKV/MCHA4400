#include <filesystem>
#include <string>
#include <iostream>
#include <cassert>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "Camera.h"
#include "Plot.h"
#include "SLAMProcessor.h"

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
        std::cout << "Export will be saved to: " << outputPath.string() << std::endl;
    }

    // Load camera calibration
    Camera camera;
    if (!camera.load(cameraPath)) {
        std::cout << "ERROR: Failed to load camera calibration from " << cameraPath.string() << std::endl;
        return;
    }

    // Display loaded calibration data
    std::cout << "✓ Camera calibration loaded successfully:" << std::endl;
    std::cout << "  Image size: " << camera.imageSize.width << "x" << camera.imageSize.height << std::endl;
    std::cout << "  Focal length: fx=" << camera.cameraMatrix.at<double>(0,0) 
              << ", fy=" << camera.cameraMatrix.at<double>(1,1) << std::endl;

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "✓ Video opened: " << nFrames << " frames at " << fps << " FPS" << std::endl;

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    // Initialize buffered video writer
    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize;
        frameSize.width     = 2*cap.get(cv::CAP_PROP_FRAME_WIDTH);  // Dual pane width
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double outputFps    = fps;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        assert(videoOut.isOpened());
        bufferedVideoWriter.start(videoOut);
        std::cout << "✓ Video export initialized (" << frameSize.width << "x" << frameSize.height << ")" << std::endl;
    }

    // Visual navigation

    // Initialisation
    SLAMProcessor slamProcessor(scenario, camera);
    Plot plot(camera);
    
    int frameCount = 0;
    std::cout << "\n=== Processing Video Frames ===" << std::endl;

    while (true)
    {
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty())
        {
            std::cout << "✓ End of video reached" << std::endl;
            break;
        }
        
        frameCount++;
        bool isLastFrame = (frameCount >= nFrames);

        // Process frame
        slamProcessor.processFrame(imgin);

        // Update state
        auto features = slamProcessor.getDetectedFeatures();
        auto featureStatus = slamProcessor.getFeatureStatus();
        auto cameraPos = slamProcessor.getCameraPosition();
        auto cameraRot = slamProcessor.getCameraRotation(); 
        auto cameraCov = slamProcessor.getCameraCovariance();
        auto landmarks = slamProcessor.getLandmarkPositions();
        auto landmarkCovs = slamProcessor.getLandmarkCovariances();
        auto landmarkStatus = slamProcessor.getLandmarkStatus();

        // Update plot
        plot.updateImage(imgin, features, featureStatus);
        plot.updateScene(cameraPos, cameraRot, cameraCov, landmarks, landmarkCovs, landmarkStatus);
        plot.render();
        
        // Handle interactivity
        plot.handleInteractivity(interactive, isLastFrame);

        // Write output frame 
        if (doExport)
        {
            cv::Mat imgout = plot.getFrame();
            assert(!imgout.empty());
            bufferedVideoWriter.write(imgout);
        }
        
        // Progress feedback
        if (frameCount % 25 == 0 || isLastFrame) {
            std::cout << "Frame " << frameCount << "/" << nFrames 
                      << " - " << features.size() << " features, "
                      << landmarks.size() << " landmarks" << std::endl;
        }
    }

    if (doExport)
    {
         bufferedVideoWriter.stop();
         std::cout << "✓ Video export completed: " << outputPath.string() << std::endl;
    }
    bufferedVideoReader.stop();
    
    std::cout << "\n=== Visual Navigation Complete ===" << std::endl;
    std::cout << "Successfully processed " << frameCount << " frames" << std::endl;
    
    // Final interactive session
    if (interactive == 1) {
        std::cout << "\nFinal interactive session active - use mouse to explore 3D view" << std::endl;
        std::cout << "Press 'q' in VTK window to exit" << std::endl;
    }
}