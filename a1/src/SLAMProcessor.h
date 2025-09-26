#ifndef SLAMPROCESSOR_H
#define SLAMPROCESSOR_H

#include <vector>
#include <random>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Core>
#include "Camera.h"

/**
 * @brief SLAM processor for all three assignment scenarios
 * 
 * Integrates Lab 2 (ArUco/corners) and Lab 3 (duck detection) foundations
 * Provides clean interface for visual navigation main loop
 */
class SLAMProcessor {
public:
    explicit SLAMProcessor(int scenario, const Camera& camera);
    ~SLAMProcessor() = default;
    
    /**
     * @brief Process a single video frame
     * @param frame Input camera frame
     */
    void processFrame(const cv::Mat& frame);
    
    // Getters for visualization (const interface)
    std::vector<cv::Point2f> getDetectedFeatures() const { return detectedFeatures; }
    std::vector<int> getFeatureStatus() const { return featureStatus; }
    Eigen::Vector3d getCameraPosition() const { return cameraPosition; }
    Eigen::Matrix3d getCameraRotation() const { return cameraRotation; }
    Eigen::Matrix3d getCameraCovariance() const { return cameraCovariance; }
    std::vector<Eigen::Vector3d> getLandmarkPositions() const { return landmarkPositions; }
    std::vector<Eigen::Matrix3d> getLandmarkCovariances() const { return landmarkCovariances; }
    std::vector<int> getLandmarkStatus() const { return landmarkStatus; }
    
    int getFrameCount() const { return frameCount; }
    int getScenario() const { return scenario; }

private:
    void initializeSLAM();
    void updateSLAMState();
    
    // Scenario-specific processing (to be extended with real lab implementations)
    void processArUcoSLAM();    // TODO: Integrate Lab 2 ArUco detection
    void processDuckSLAM();     // TODO: Integrate Lab 3 duck detection  
    void processPointSLAM();    // TODO: Integrate Lab 2 corner detection
    
    // Core data
    int scenario;
    Camera camera;
    int frameCount;
    cv::Mat currentFrame;
    std::mt19937 rng;
    
    // Detection results (for visualization)
    std::vector<cv::Point2f> detectedFeatures;
    std::vector<int> featureStatus; // 0=tracked, 1=visible_undetected
    
    // SLAM state estimates
    Eigen::Vector3d cameraPosition;
    Eigen::Matrix3d cameraRotation;
    Eigen::Matrix3d cameraCovariance;
    std::vector<Eigen::Vector3d> landmarkPositions;
    std::vector<Eigen::Matrix3d> landmarkCovariances;
    std::vector<int> landmarkStatus; // 0=tracked, 1=visible_undetected, 2=not_visible
    
    // Detector instances (Lab foundations)
    cv::Ptr<cv::aruco::Dictionary> arucoDict;
    cv::Ptr<cv::aruco::DetectorParameters> arucoParams;
};

#endif // SLAMPROCESSOR_H