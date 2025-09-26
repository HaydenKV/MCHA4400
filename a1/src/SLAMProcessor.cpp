// SLAMProcessor.cpp - Fixed ArUco initialization for OpenCV 4.7+
#include "SLAMProcessor.h"
#include <iostream>
#include <opencv2/aruco.hpp>

SLAMProcessor::SLAMProcessor(int scenario, const Camera& camera) 
    : scenario(scenario), camera(camera), frameCount(0), rng(std::random_device{}()) {
    std::cout << "Initializing SLAM processor for scenario " << scenario << std::endl;
    
    switch(scenario) {
        case 1: std::cout << "  -> Tag SLAM (ArUco markers)" << std::endl; break;
        case 2: std::cout << "  -> Duck SLAM (identical objects)" << std::endl; break;
        case 3: std::cout << "  -> Point SLAM (corner features)" << std::endl; break;
        default: std::cout << "  -> Unknown scenario " << scenario 
                          << std::endl; break;
    }
    
    initializeSLAM();
}

void SLAMProcessor::initializeSLAM() {
    // Initialize camera state
    cameraPosition = Eigen::Vector3d(0, 0, 0);
    cameraRotation = Eigen::Matrix3d::Identity();
    cameraCovariance = 0.1 * Eigen::Matrix3d::Identity();
    
    // Initialize landmarks in a realistic pattern
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            Eigen::Vector3d pos(i * 0.8 - 0.8, j * 0.8 - 0.4, 0.2);
            landmarkPositions.push_back(pos);
            
            Eigen::Matrix3d cov = 0.05 * Eigen::Matrix3d::Identity();
            landmarkCovariances.push_back(cov);
            
            landmarkStatus.push_back(0); // Start as tracked
        }
    }
    
    // Initialize scenario-specific detectors
    if (scenario == 1) {
        // ArUco detector setup (OpenCV 4.7+ API from Lab 2)
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
        arucoDetector = cv::makePtr<cv::aruco::ArucoDetector>(dictionary, detectorParams);
        std::cout << "  -> ArUco detector initialized" << std::endl;
    }
    // scenario 2 and 3 detector initialization would go here
}

void SLAMProcessor::processFrame(const cv::Mat& frame) {
    frameCount++;
    currentFrame = frame.clone();
    
    // Clear previous detections
    detectedFeatures.clear();
    featureStatus.clear();
    
    // Process based on scenario
    switch(scenario) {
        case 1: processArUcoSLAM(); break;
        case 2: processDuckSLAM(); break;
        case 3: processPointSLAM(); break;
        default:
            std::cout << "Warning: Unknown scenario " << scenario << std::endl;
            break;
    }
    
    // Update SLAM state estimates
    updateSLAMState();
}

void SLAMProcessor::processArUcoSLAM() {
    // ArUco detection using OpenCV 4.7+ API (based on Lab 2 implementation)
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    
    // Use the new detector interface
    arucoDetector->detectMarkers(currentFrame, markerCorners, markerIds);
    
    // Process detected markers
    for (size_t i = 0; i < markerIds.size(); i++) {
        // Add all four corners of each marker as features
        for (const auto& corner : markerCorners[i]) {
            detectedFeatures.push_back(corner);
            featureStatus.push_back(0); // Tracked (blue in visualization)
        }
        
        // TODO: 
        // 1. Estimate 6-DOF marker pose using cv::aruco::estimatePoseSingleMarkers
        // 2. Update corresponding landmark in SLAM map
        // 3. Perform data association based on marker IDs
        // 4. Apply Lab 9 geometric compatibility matching if needed
    }
}

void SLAMProcessor::processDuckSLAM() {
    // TODO: Integrate Lab 3 duck detection with ONNX model
    // For now, simulate some detections for visualization
    if (frameCount % 10 == 0) {
        // Add simulated duck detection
        cv::Point2f duckCenter(320 + (rng() % 200 - 100), 240 + (rng() % 150 - 75));
        detectedFeatures.push_back(duckCenter);
        featureStatus.push_back(0); // Tracked
    }
}

void SLAMProcessor::processPointSLAM() {
    // TODO: Integrate Lab 2 corner detection (FAST, Harris, Shi-Tomasi)
    // For now, simulate some corner detections
    for (int i = 0; i < 5; i++) {
        cv::Point2f corner(100 + i * 120 + (rng() % 40 - 20), 
                          200 + (rng() % 100 - 50));
        detectedFeatures.push_back(corner);
        featureStatus.push_back(i % 2); // Mix of tracked and undetected
    }
}

void SLAMProcessor::updateSLAMState() {
    // TODO: Implement SLAM filtering based on Labs 6-7 (Laplace filters)
    // For now, add some realistic motion simulation
    
    // Simple camera motion simulation
    double dt = 0.033; // ~30fps
    Eigen::Vector3d velocity(0.01, 0, 0.005);
    cameraPosition += velocity * dt;
    
    // Add some noise to covariance to simulate uncertainty growth
    cameraCovariance += 0.001 * Eigen::Matrix3d::Identity();
    
    // Update landmark uncertainties
    for (auto& cov : landmarkCovariances) {
        cov += 0.0001 * Eigen::Matrix3d::Identity();
    }
    
    // Simulate landmark status changes based on visibility
    for (size_t i = 0; i < landmarkStatus.size(); i++) {
        if (rng() % 100 < 5) { // 5% chance to change status
            landmarkStatus[i] = (landmarkStatus[i] + 1) % 3;
        }
    }
}