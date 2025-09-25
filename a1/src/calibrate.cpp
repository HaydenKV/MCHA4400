#include <filesystem>
#include "Camera.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "calibrate.h"

void calibrateCamera(const std::filesystem::path & configPath)
{
    // TODO
    // - Read XML at configPath
    // - Parse XML and extract relevant frames from source video containing the chessboard
    // - Perform camera calibration
    // - Write the camera matrix and lens distortion parameters to camera.xml file in same directory as configPath
    // - Visualise the camera calibration results
    
    std::cout << "Loading calibration configuration from: " << configPath.string() << std::endl;
    
    // Check if configuration exists
    if (!std::filesystem::exists(configPath))
    {
        std::cerr << "Configuration file does not exist: " << configPath.string() << std::endl;
        return;
    }
    
    // Load chessboard data from configuration (this handles video parsing)
    ChessboardData chessboardData(configPath);
    
    if (chessboardData.chessboardImages.empty())
    {
        std::cerr << "No chessboard images found for calibration" << std::endl;
        return;
    }
    
    std::cout << "Found " << chessboardData.chessboardImages.size() 
              << " images with detected chessboards" << std::endl;
    
    // Create camera object and calibrate
    Camera camera;
    camera.calibrate(chessboardData);
    
    // Write calibration to camera.xml in same directory as config
    std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";
    std::cout << "Writing camera calibration to: " << cameraPath.string() << std::endl;
    
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::WRITE);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open file for writing: " << cameraPath.string() << std::endl;
        return;
    }
    
    fs << "camera" << camera;
    fs.release();
    
    std::cout << "Camera calibration completed successfully!" << std::endl;
    
    // Optional: Display validation visualization
    // Show some calibration images with detected corners
    chessboardData.drawCorners();
    chessboardData.drawBoxes(camera);
    
    std::cout << "Press any key in image window to continue through validation images..." << std::endl;
    
    // Display first few images for validation
    int maxDisplay = std::min(5, (int)chessboardData.chessboardImages.size());
    for (int i = 0; i < maxDisplay; i++)
    {
        cv::imshow("Calibration Validation", chessboardData.chessboardImages[i].image);
        cv::waitKey(0); // Wait for key press
    }
    
    cv::destroyAllWindows();
}
